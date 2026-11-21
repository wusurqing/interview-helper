"""索引调度器：扫 kb_dir → dispatch loader → chunker → store。"""
from __future__ import annotations

import os
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Optional

from core.config import get_config
from core.logger import get_logger
from services.storage.paths import backend_root

from . import loaders as _loaders  # noqa: F401  触发 register 副作用
from .chunker import chunk_doc
from .store import KBStore

_log = get_logger(__name__)

_store: Optional[KBStore] = None
_lock = threading.RLock()

_FAILED_DOC_EXTENSIONS = {".doc"}
_DOC_FAIL_MSG = "不支持 .doc，请在 Word 里另存为 .docx 后再上传"


def resolve_path(path_str: str) -> Path:
    """相对路径按 backend_root 解析; 绝对路径原样返回。"""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return Path(backend_root()) / p


def validate_rel_path(rel_path: str) -> str:
    """校验 KB 内相对路径并返回以 ``/`` 分隔的规范形式；非法时抛 ``ValueError``。

    拒绝：空串、控制字符、任何 ``:``（Windows 驱动器相对路径 ``C:evil``、
    NTFS 备用数据流 ``a.md:ads``）、带 drive/root 的路径（含 UNC 与以
    ``/``/``\\`` 开头）、``..``/``.``/空分段。
    """
    s = rel_path or ""
    if not s.strip():
        raise ValueError("path required")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in s):
        raise ValueError("path must not contain control characters")
    if ":" in s:
        raise ValueError("path must not contain ':'")
    win = PureWindowsPath(s)
    if win.drive or win.root or PurePosixPath(s).root:
        raise ValueError("absolute path not allowed")
    parts = s.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("path traversal detected")
    return "/".join(parts)


def resolve_kb_file(rel_path: str) -> Path:
    """把 KB 相对路径解析为 kb_dir 内的绝对路径；越界（含符号链接逃逸）抛 ``ValueError``。

    返回值为 ``<resolved kb_dir>/<rel>``（不跟随末级符号链接），调用方可直接写入或删除。
    """
    norm = validate_rel_path(rel_path)
    root = resolve_path(get_config().kb_dir).resolve()
    full = root / norm
    resolved = full.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("path escapes kb_dir")
    return full


def _resolved_db_path() -> str:
    return str(resolve_path(get_config().kb_db_path))


def _get_store() -> KBStore:
    global _store
    with _lock:
        db_path = _resolved_db_path()
        if _store is None or _store.db_path != db_path:
            _store = KBStore(db_path)
            _store.init_schema()
        return _store


def reset() -> None:
    """测试辅助：丢弃单例，下次 _get_store() 会按当前 config 重建。"""
    global _store
    with _lock:
        _store = None


def list_docs(limit: Optional[int] = None) -> list[dict[str, Any]]:
    return _get_store().list_docs(limit=limit)


def stats() -> dict[str, Any]:
    return _get_store().stats()


def _index_single_file(
    path: Path,
    rel: str,
    *,
    cfg: Any,
    store: KBStore,
    allowed: set[str],
    force: bool = False,
) -> bool:
    """索引单个文件；实际重建了 chunk 返回 True，跳过或失败返回 False。"""
    ext = path.suffix.lower()
    if ext in _FAILED_DOC_EXTENSIONS:
        try:
            st = path.stat()
            store.upsert_doc(rel, st.st_mtime, st.st_size, "unsupported", None)
            store.set_status(rel, "failed", _DOC_FAIL_MSG)
        except Exception as e:  # pragma: no cover
            _log.warning("kb indexer 记录 .doc 失败 %s: %s", rel, e)
        return False

    if ext not in allowed:
        return False

    loader = _loaders.dispatch_loader(path)
    if loader is None:
        return False

    try:
        stat = path.stat()
    except OSError as e:
        _log.warning("kb indexer stat 失败 %s: %s", rel, e)
        return False

    if not force:
        existing = store.get_doc(rel)
        if (
            existing
            and existing["mtime"] == stat.st_mtime
            and existing["size"] == stat.st_size
            and existing["status"] == "ok"
        ):
            return False

    try:
        raw = loader.load(path, rel_path=rel)
        doc_id = store.upsert_doc(
            path=rel,
            mtime=stat.st_mtime,
            size=stat.st_size,
            loader=getattr(loader, "name", "unknown"),
            title=raw.title,
        )
        chunks = chunk_doc(raw, max_chars=cfg.kb_chunk_max_chars)
        store.replace_chunks(doc_id, chunks)
        return True
    except Exception as e:
        _log.warning("kb indexer 加载 %s 失败：%s", rel, e)
        try:
            store.upsert_doc(
                rel,
                stat.st_mtime,
                stat.st_size,
                getattr(loader, "name", "unknown"),
                None,
            )
            store.set_status(rel, "failed", str(e)[:500])
        except Exception as inner:  # pragma: no cover
            _log.warning("kb indexer 记录失败状态时再次失败 %s: %s", rel, inner)
        return False


def reindex(force: bool = False) -> dict[str, Any]:
    cfg = get_config()
    kb_dir = resolve_path(cfg.kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)
    store = _get_store()
    allowed = {ext.lower() for ext in cfg.kb_file_extensions}
    reindexed = 0
    on_disk: set[str] = set()

    for path in sorted(kb_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = str(path.relative_to(kb_dir)).replace(os.sep, "/")
        except ValueError:
            continue
        on_disk.add(rel)

        if _index_single_file(
            path,
            rel,
            cfg=cfg,
            store=store,
            allowed=allowed,
            force=force,
        ):
            reindexed += 1

    for row in store.list_docs():
        if row["path"] not in on_disk:
            store.delete_doc(row["path"])

    info = store.stats()
    info["reindexed_files"] = reindexed
    return info


def reindex_file(rel_path: str, *, force: bool = True) -> dict[str, Any]:
    """按相对路径增量重建单文件索引（不再触发全库 rglob 扫描）；文件不存在时删除记录。"""
    norm = validate_rel_path(rel_path)
    path = resolve_kb_file(norm)
    store = _get_store()
    if not path.exists() or not path.is_file():
        store.delete_doc(norm)
        info = store.stats()
        info["reindexed_files"] = 0
        return info

    cfg = get_config()
    allowed = {ext.lower() for ext in cfg.kb_file_extensions}
    reindexed = (
        1
        if _index_single_file(
            path,
            norm,
            cfg=cfg,
            store=store,
            allowed=allowed,
            force=force,
        )
        else 0
    )
    info = store.stats()
    info["reindexed_files"] = reindexed
    return info


def remove_file(rel_path: str) -> None:
    """删除索引 + 磁盘文件。非法路径（越出 kb_dir）直接拒绝、不做任何删除；其余失败不抛。"""
    try:
        full = resolve_kb_file(rel_path)
    except ValueError as e:
        _log.warning("kb indexer remove_file 拒绝非法路径 %r: %s", rel_path, e)
        return
    store = _get_store()
    store.delete_doc(validate_rel_path(rel_path))
    try:
        if full.is_file() or full.is_symlink():
            full.unlink()
    except Exception as e:  # pragma: no cover
        _log.warning("kb indexer remove_file 删除磁盘文件失败 %s: %s", rel_path, e)


__all__ = [
    "reset",
    "list_docs",
    "stats",
    "reindex",
    "reindex_file",
    "remove_file",
    "resolve_path",
    "resolve_kb_file",
    "validate_rel_path",
]
