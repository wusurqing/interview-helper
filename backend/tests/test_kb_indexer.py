from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture
def setup_kb(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    db = tmp_path / "kb.sqlite"

    from core import config as cfg_mod
    cfg = cfg_mod.get_config()
    monkeypatch.setattr(cfg, "kb_dir", str(kb_dir), raising=False)
    monkeypatch.setattr(cfg, "kb_db_path", str(db), raising=False)
    monkeypatch.setattr(cfg, "kb_cache_dir", str(tmp_path / "kb_cache"), raising=False)

    from services.kb import indexer
    indexer.reset()
    yield indexer, kb_dir, db
    indexer.reset()


def test_reindex_creates_schema_and_indexes_md(setup_kb):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "redis.md").write_text("# Redis\n\nRedis hits", encoding="utf-8")
    info = indexer.reindex()
    assert info["total_docs"] == 1
    assert info["total_chunks"] >= 1
    assert info["reindexed_files"] == 1


def test_reindex_skips_unchanged_file(setup_kb):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "a.md").write_text("# A\n\ntext", encoding="utf-8")
    indexer.reindex()
    info2 = indexer.reindex()
    assert info2["reindexed_files"] == 0
    assert info2["total_docs"] == 1


def test_reindex_reruns_on_mtime_change(setup_kb):
    indexer, kb_dir, _ = setup_kb
    f = kb_dir / "a.md"
    f.write_text("# A\n\nfirst", encoding="utf-8")
    indexer.reindex()
    # 改内容 + 改 mtime
    f.write_text("# A\n\nsecond version", encoding="utf-8")
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 10))
    info = indexer.reindex()
    assert info["reindexed_files"] == 1


def test_reindex_marks_doc_as_failed_on_broken_loader(setup_kb, monkeypatch):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "a.md").write_text("# A\n\nt", encoding="utf-8")

    from services.kb.loaders import markdown as md_mod

    def boom(self, file_path, *, rel_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(md_mod.MarkdownLoader, "load", boom)
    indexer.reindex()
    docs = indexer.list_docs()
    assert len(docs) == 1
    assert docs[0]["status"] == "failed"
    assert "boom" in (docs[0]["error"] or "")


def test_doc_file_rejected_with_clear_message(setup_kb):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "legacy.doc").write_bytes(b"fake")
    indexer.reindex()
    docs = {d["path"]: d for d in indexer.list_docs()}
    assert "legacy.doc" in docs
    assert docs["legacy.doc"]["status"] == "failed"
    assert ".docx" in (docs["legacy.doc"]["error"] or "")


def test_remove_file_drops_doc(setup_kb):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "a.md").write_text("# A\n\nt", encoding="utf-8")
    indexer.reindex()
    os.remove(kb_dir / "a.md")
    indexer.reindex()
    assert indexer.list_docs() == []


def test_unknown_extension_silently_skipped(setup_kb):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "weird.xyz").write_text("abc", encoding="utf-8")
    (kb_dir / "ok.md").write_text("# A\n\nt", encoding="utf-8")
    info = indexer.reindex()
    assert info["total_docs"] == 1  # weird.xyz 不应进库


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a.md", "a.md"),
        ("sub/a.md", "sub/a.md"),
        ("sub\\deep\\a.md", "sub/deep/a.md"),
    ],
)
def test_validate_rel_path_accepts_and_normalizes(setup_kb, raw, expected):
    indexer, _, _ = setup_kb
    assert indexer.validate_rel_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "C:evil.md",
        "C:evil/a.md",
        "D:foo/bar.md",
        "C:\\abs.md",
        "\\\\server\\share\\a.md",
        "//server/share/a.md",
        "/etc/passwd",
        "\\rooted.md",
        "a.md:ads",
        "../x.md",
        "sub/../../x.md",
        "sub/./a.md",
        "sub//a.md",
        "sub/",
        "a\x00.md",
        "a\x1f.md",
        "a\x7f.md",
    ],
)
def test_validate_rel_path_rejects_escape_vectors(setup_kb, raw):
    indexer, _, _ = setup_kb
    with pytest.raises(ValueError):
        indexer.validate_rel_path(raw)
    with pytest.raises(ValueError):
        indexer.resolve_kb_file(raw)


def test_resolve_kb_file_stays_inside_kb_dir(setup_kb):
    indexer, kb_dir, _ = setup_kb
    full = indexer.resolve_kb_file("sub/a.md")
    assert full == kb_dir.resolve() / "sub" / "a.md"


def _make_dir_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    pytest.skip("cannot create directory link on this platform")


def test_remove_file_refuses_symlink_escape(setup_kb, tmp_path):
    indexer, kb_dir, _ = setup_kb
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.md"
    victim.write_text("keep", encoding="utf-8")
    _make_dir_link(kb_dir / "link", outside)

    with pytest.raises(ValueError):
        indexer.resolve_kb_file("link/victim.md")
    indexer.remove_file("link/victim.md")  # 不抛、也不删除
    assert victim.exists()


def test_remove_file_refuses_absolute_path(setup_kb, tmp_path):
    indexer, _, _ = setup_kb
    victim = tmp_path / "victim.md"
    victim.write_text("keep", encoding="utf-8")
    indexer.remove_file(str(victim))
    assert victim.exists()


def test_remove_file_deletes_in_kb_file_and_index(setup_kb):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "sub").mkdir()
    (kb_dir / "sub" / "a.md").write_text("# A\n\nt", encoding="utf-8")
    indexer.reindex()
    assert any(d["path"] == "sub/a.md" for d in indexer.list_docs())
    indexer.remove_file("sub\\a.md")
    assert not (kb_dir / "sub" / "a.md").exists()
    assert indexer.list_docs() == []


def test_reindex_file_rejects_escape(setup_kb):
    indexer, _, _ = setup_kb
    with pytest.raises(ValueError):
        indexer.reindex_file("C:evil.md")


def test_cli_list_outputs_indexed_docs(setup_kb, capsys):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "a.md").write_text("# A\n\nt", encoding="utf-8")
    indexer.reindex()
    from services.kb import cli
    exit_code = cli.main(["list"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "a.md" in out
    assert "markdown" in out


def test_cli_reindex_outputs_stats(setup_kb, capsys):
    indexer, kb_dir, _ = setup_kb
    (kb_dir / "a.md").write_text("# A\n\nt", encoding="utf-8")
    from services.kb import cli
    exit_code = cli.main(["reindex"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "total_docs" in out


def test_remove_file_during_reindex_does_not_resurrect_ghost_doc(setup_kb, monkeypatch):
    """reindex 已扫描并正在加载某文件时并发删除：删除必须等重建结束，不能被重建写回幽灵文档。"""
    import threading

    indexer, kb_dir, _ = setup_kb
    (kb_dir / "a.md").write_text("# A\n\nalpha", encoding="utf-8")

    from services.kb.loaders import markdown as md_mod

    real_load = md_mod.MarkdownLoader.load
    loading = threading.Event()
    release = threading.Event()

    def slow_load(self, file_path, *, rel_path):
        raw = real_load(self, file_path, rel_path=rel_path)
        loading.set()
        assert release.wait(timeout=5.0)
        return raw

    monkeypatch.setattr(md_mod.MarkdownLoader, "load", slow_load)

    errors: list[BaseException] = []

    def run(fn, *args):
        try:
            fn(*args)
        except BaseException as e:  # pragma: no cover - surfaced via assert below
            errors.append(e)

    t_reindex = threading.Thread(target=run, args=(indexer.reindex, True))
    t_reindex.start()
    try:
        assert loading.wait(timeout=5.0)
        t_remove = threading.Thread(target=run, args=(indexer.remove_file, "a.md"))
        t_remove.start()
        # 重建持有互斥期间删除不得完成（否则重建随后会把记录写回）
        t_remove.join(timeout=0.3)
        assert t_remove.is_alive()
    finally:
        release.set()
    t_reindex.join(timeout=5.0)
    t_remove.join(timeout=5.0)
    assert not t_reindex.is_alive() and not t_remove.is_alive()
    assert errors == []

    assert not (kb_dir / "a.md").exists()
    assert indexer.list_docs() == []
    assert indexer.stats()["total_chunks"] == 0


def test_kb_delete_route_runs_on_index_lane(setup_kb, monkeypatch):
    """删除接口必须与 reindex/reindex_file 同走 KB 索引泳道，而不是查询泳道。"""
    import threading

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import kb as kb_api
    from core import resource_lanes as rl

    indexer, kb_dir, _ = setup_kb
    (kb_dir / "a.md").write_text("# A\n\nt", encoding="utf-8")
    seen: dict[str, str] = {}
    real_remove = indexer.remove_file

    def spy_remove(rel):
        seen["thread"] = threading.current_thread().name
        return real_remove(rel)

    monkeypatch.setattr(indexer, "remove_file", spy_remove)

    app = FastAPI()
    app.include_router(kb_api.router, prefix="/api")
    try:
        with TestClient(app) as client:
            res = client.delete("/api/kb/docs", params={"path": "a.md"})
        assert res.status_code == 200
        assert seen["thread"].startswith("ia-kb-index")
        assert not (kb_dir / "a.md").exists()
    finally:
        rl.shutdown_resource_lanes(wait=True)
