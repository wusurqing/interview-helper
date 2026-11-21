"""PDF loader：L1 文本层（pypdf），L2 OCR / L3 Vision 均在索引期、opt-in。"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import shutil
from typing import Callable, Optional

from core.logger import get_logger

from ..types import RawDoc, RawSection
from ._base import register
from ._ocr import ocr_image_file
from ._vision import render_pdf_page_to_png, vision_caption

_log = get_logger(__name__)

try:
    from pypdf import PdfReader
    _PYPDF_OK = True
except Exception as e:  # pragma: no cover
    _log.info("pypdf not available: %s", e)
    _PYPDF_OK = False

_EMPTY_PAGE_THRESHOLD = 100
MAX_PDF_PAGES = 200


def _l2_ocr_enabled() -> bool:
    try:
        from core.config import get_config  # type: ignore
        return bool(getattr(get_config(), "kb_ocr_enabled", False))
    except Exception:  # pragma: no cover
        return False


def _l3_vision_enabled() -> bool:
    try:
        from core.config import get_config  # type: ignore
        return bool(getattr(get_config(), "kb_vision_caption_enabled", False))
    except Exception:  # pragma: no cover
        return False


def _max_pdf_pages() -> int:
    try:
        from core.config import get_config  # type: ignore
        configured = getattr(get_config(), "kb_pdf_max_pages", None)
        if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
            return configured
    except Exception:  # pragma: no cover
        pass
    return MAX_PDF_PAGES


def _kb_cache_root() -> Path:
    try:
        from core.config import get_config  # type: ignore
        from services.kb.indexer import resolve_path  # type: ignore
        return resolve_path(getattr(get_config(), "kb_cache_dir", "data/kb_cache"))
    except Exception:  # pragma: no cover
        return Path("data/kb_cache")


def cache_key_for_rel_path(rel_path: str) -> str:
    """Stable hash of normalized KB relative path so same-named PDFs in different subdirs never collide."""
    norm = (rel_path or "").replace("\\", "/").strip("/")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def cache_dirs_for_rel_path(rel_path: str) -> tuple[Path, Path]:
    key = cache_key_for_rel_path(rel_path)
    root = _kb_cache_root()
    return root / "ocr" / key, root / "vision" / key


def clear_pdf_cache(rel_path: str) -> None:
    """Remove cached OCR/Vision page renders for a KB document (best-effort)."""
    if not rel_path:
        return
    for cache_dir in cache_dirs_for_rel_path(rel_path):
        try:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        except OSError as e:  # pragma: no cover
            _log.warning("kb 清理 PDF 缓存失败 %s: %s", cache_dir, e)


def _page_cache_key(pdf_path: Path, rel_path: Optional[str]) -> str:
    if rel_path:
        return cache_key_for_rel_path(rel_path)
    return cache_key_for_rel_path(str(pdf_path.resolve()))


def _run_ocr_on_pdf_page(pdf_path: Path, page_idx: int, rel_path: Optional[str] = None) -> str:
    out = _kb_cache_root() / "ocr" / _page_cache_key(pdf_path, rel_path) / f"page_{page_idx}.png"
    if not render_pdf_page_to_png(pdf_path, page_idx, out):
        return ""
    return ocr_image_file(out)


def _run_vision_caption(pdf_path: Path, page_idx: int, rel_path: Optional[str] = None) -> str:
    out = _kb_cache_root() / "vision" / _page_cache_key(pdf_path, rel_path) / f"page_{page_idx}.png"
    if not render_pdf_page_to_png(pdf_path, page_idx, out):
        return ""
    cap = vision_caption(out)
    return cap or ""


def _invoke_page_runner(
    fn: Callable[..., str],
    pdf_path: Path,
    page_idx: int,
    rel_path: str,
) -> str:
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        accepts_rel = (
            "rel_path" in sig.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            or sum(
                1
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ) >= 3
        )
    except (TypeError, ValueError):
        accepts_rel = False
    if accepts_rel:
        return fn(pdf_path, page_idx, rel_path=rel_path)
    return fn(pdf_path, page_idx)


class PdfLoader:
    name = "pdf"
    extensions = (".pdf",)

    def load(self, file_path: Path, *, rel_path: str) -> RawDoc:
        if not _PYPDF_OK:
            raise RuntimeError("pypdf 未安装：请 `pip install pypdf`")
        reader = PdfReader(str(file_path))
        page_count = len(reader.pages)
        max_pages = _max_pdf_pages()
        if page_count > max_pages:
            raise ValueError(f"PDF 页数过多（{page_count} 页，上限 {max_pages} 页）")
        title = _pdf_title(reader, rel_path)
        sections: list[RawSection] = []
        for idx, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as e:
                _log.warning("pypdf extract_text page %d 失败: %s", idx, e)
                text = ""
            if text:
                sections.append(
                    RawSection(
                        section_path=f"{title} > Page {idx}",
                        text=text,
                        page=idx,
                        origin="text",
                    )
                )
            is_image_page = len(text) < _EMPTY_PAGE_THRESHOLD
            if is_image_page and _l2_ocr_enabled():
                try:
                    ocr_text = _invoke_page_runner(_run_ocr_on_pdf_page, file_path, idx, rel_path)
                except Exception as e:  # pragma: no cover
                    _log.warning("kb OCR 调用异常 %s p%d: %s", rel_path, idx, e)
                    ocr_text = ""
                if ocr_text:
                    sections.append(
                        RawSection(
                            section_path=f"{title} > Page {idx} (OCR)",
                            text=ocr_text,
                            page=idx,
                            origin="ocr",
                        )
                    )
            if _l3_vision_enabled():
                try:
                    cap = _invoke_page_runner(_run_vision_caption, file_path, idx, rel_path)
                except Exception as e:  # pragma: no cover
                    _log.warning("kb Vision 调用异常 %s p%d: %s", rel_path, idx, e)
                    cap = ""
                if cap:
                    sections.append(
                        RawSection(
                            section_path=f"{title} > Page {idx} (Vision)",
                            text=cap,
                            page=idx,
                            origin="vision",
                        )
                    )
        return RawDoc(path=rel_path, title=title, loader="pdf", sections=sections)


def _pdf_title(reader: "PdfReader", rel_path: str) -> str:
    try:
        meta = reader.metadata
        if meta and getattr(meta, "title", None):
            t = str(meta.title).strip()
            if t:
                return t
    except Exception:
        pass
    return Path(rel_path).stem


if _PYPDF_OK:
    register(PdfLoader())

__all__ = ["PdfLoader", "cache_key_for_rel_path", "cache_dirs_for_rel_path", "clear_pdf_cache"]
