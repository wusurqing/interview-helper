"""Tests for DOCX Zip-bomb guards, PDF page caps, KB cache hashing/cleanup, and KB vision model selection."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from core import config as cfg_mod
from services import resume
from services.kb import indexer
from services.kb.loaders import _vision as kb_vision
from services.kb.loaders import docx as kb_docx
from services.kb.loaders import pdf as kb_pdf


def _make_minimal_docx_bytes(text: str = "hello resume") -> bytes:
    buf = BytesIO()
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))
    return buf.getvalue()


def _make_pdf_file(path: Path, pages: int, text_prefix: str = "page") -> None:
    import fitz

    doc = fitz.open()
    try:
        for idx in range(1, pages + 1):
            page = doc.new_page()
            page.insert_text((72, 72), f"{text_prefix} {idx}")
        doc.save(str(path))
    finally:
        doc.close()


def test_resume_docx_parses_valid_and_rejects_bad_zip():
    assert resume._parse_docx(_make_minimal_docx_bytes("Alice Engineer")) == "Alice Engineer"
    with pytest.raises(ValueError, match="无效的 DOCX"):
        resume._parse_docx(b"not-a-zip")


def test_resume_docx_rejects_entry_count_and_size_bombs(monkeypatch):
    monkeypatch.setattr(resume, "MAX_DOCX_ZIP_ENTRIES", 3)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"<w:document/>")
        for i in range(4):
            zf.writestr(f"extra_{i}.txt", b"x")
    with pytest.raises(ValueError, match="条目数"):
        resume._parse_docx(buf.getvalue())

    monkeypatch.setattr(resume, "MAX_DOCX_ZIP_ENTRIES", 100)
    monkeypatch.setattr(resume, "MAX_DOCX_UNCOMPRESSED_BYTES", 512)
    big_buf = BytesIO()
    with zipfile.ZipFile(big_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"a" * 1024)
    with pytest.raises(ValueError, match="体积超过"):
        resume._parse_docx(big_buf.getvalue())

    monkeypatch.setattr(resume, "MAX_DOCX_UNCOMPRESSED_BYTES", 10_000)
    monkeypatch.setattr(resume, "MAX_DOCX_XML_BYTES", 200)
    with pytest.raises(ValueError, match="XML"):
        resume._parse_docx(big_buf.getvalue())


def test_resume_pdf_enforces_max_pages(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "long_resume.pdf"
    _make_pdf_file(pdf_path, pages=4)
    monkeypatch.setattr(resume, "MAX_RESUME_PDF_PAGES", 3)
    with pytest.raises(ValueError, match="页数过多"):
        resume._pdf_to_images_base64(str(pdf_path))


def test_kb_docx_loader_rejects_zip_bombs(tmp_path: Path, monkeypatch):
    pytest.importorskip("docx")
    loader = kb_docx.DocxLoader()

    many_entries = tmp_path / "many.docx"
    with zipfile.ZipFile(many_entries, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(6):
            zf.writestr(f"f_{i}.xml", b"<x/>")
    monkeypatch.setattr(kb_docx, "MAX_DOCX_ZIP_ENTRIES", 5)
    with pytest.raises(ValueError, match="条目数"):
        loader.load(many_entries, rel_path="many.docx")

    monkeypatch.setattr(kb_docx, "MAX_DOCX_ZIP_ENTRIES", 100)
    monkeypatch.setattr(kb_docx, "MAX_DOCX_UNCOMPRESSED_BYTES", 256)
    huge = tmp_path / "huge.docx"
    with zipfile.ZipFile(huge, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", b"A" * 1024)
    with pytest.raises(ValueError, match="体积超过"):
        loader.load(huge, rel_path="huge.docx")

    corrupt = tmp_path / "corrupt.docx"
    corrupt.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="无效的 DOCX"):
        loader.load(corrupt, rel_path="corrupt.docx")


def test_kb_pdf_loader_enforces_max_pages(tmp_path: Path, monkeypatch):
    pytest.importorskip("pypdf")
    pdf_path = tmp_path / "big.pdf"
    _make_pdf_file(pdf_path, pages=5)
    monkeypatch.setattr(kb_pdf, "MAX_PDF_PAGES", 3)
    loader = kb_pdf.PdfLoader()
    with pytest.raises(ValueError, match="页数过多"):
        loader.load(pdf_path, rel_path="big.pdf")


def test_kb_pdf_cache_uses_rel_path_hash_and_cleans_up_on_remove(tmp_path: Path, monkeypatch):
    pytest.importorskip("pypdf")
    kb_dir = tmp_path / "kb"
    cache_dir = tmp_path / "kb_cache"
    db_path = tmp_path / "kb.sqlite"
    (kb_dir / "dir_a").mkdir(parents=True)
    (kb_dir / "dir_b").mkdir(parents=True)

    pdf_a = kb_dir / "dir_a" / "notes.pdf"
    pdf_b = kb_dir / "dir_b" / "notes.pdf"
    _make_pdf_file(pdf_a, pages=1, text_prefix="a")
    _make_pdf_file(pdf_b, pages=1, text_prefix="b")

    cfg = cfg_mod.get_config()
    monkeypatch.setattr(cfg, "kb_dir", str(kb_dir), raising=False)
    monkeypatch.setattr(cfg, "kb_db_path", str(db_path), raising=False)
    monkeypatch.setattr(cfg, "kb_cache_dir", str(cache_dir), raising=False)
    monkeypatch.setattr(cfg, "kb_ocr_enabled", True, raising=False)
    monkeypatch.setattr(cfg, "kb_vision_caption_enabled", True, raising=False)
    indexer.reset()

    monkeypatch.setattr(kb_pdf, "ocr_image_file", lambda p: f"ocr:{p.parent.name}")
    monkeypatch.setattr(kb_pdf, "vision_caption", lambda p: f"vision:{p.parent.name}")

    indexer.reindex(force=True)

    ocr_a, vis_a = kb_pdf.cache_dirs_for_rel_path("dir_a/notes.pdf")
    ocr_b, vis_b = kb_pdf.cache_dirs_for_rel_path("dir_b/notes.pdf")
    assert ocr_a != ocr_b
    assert vis_a != vis_b
    assert (ocr_a / "page_1.png").exists()
    assert (vis_a / "page_1.png").exists()
    assert (ocr_b / "page_1.png").exists()
    assert (vis_b / "page_1.png").exists()

    # Removing dir_a/notes.pdf removes only its cache dirs, leaving dir_b intact
    indexer.remove_file("dir_a/notes.pdf")
    assert not ocr_a.exists()
    assert not vis_a.exists()
    assert (ocr_b / "page_1.png").exists()
    assert (vis_b / "page_1.png").exists()

    # Deleting dir_b/notes.pdf from disk and reindexing also prunes its cache
    pdf_b.unlink()
    indexer.reindex()
    assert not ocr_b.exists()
    assert not vis_b.exists()
    indexer.reset()


def test_kb_vision_prefers_selected_model_and_disables_thinking(tmp_path: Path, monkeypatch):
    m_first = cfg_mod.ModelConfig(
        id="v-first", name="FirstVision", api_key="sk-1", model="vl-1",
        supports_vision=True, supports_think=True,
        think_disabled_params={"enable_thinking": False},
    )
    m_screenshot = cfg_mod.ModelConfig(
        id="v-shot", name="ShotVision", api_key="sk-2", model="vl-2",
        supports_vision=True, supports_think=True,
        think_disabled_params={"reasoning_effort": "none"},
    )
    m_active = cfg_mod.ModelConfig(
        id="v-active", name="ActiveVision", api_key="sk-3", model="vl-3",
        supports_vision=True, supports_think=False,
    )

    cfg = SimpleNamespace(
        models=[m_first, m_screenshot, m_active],
        assist_screenshot_model_id="v-shot",
        active_model_id="v-active",
        active_model=2,
    )
    monkeypatch.setattr(cfg_mod, "get_config", lambda: cfg)
    assert kb_vision._pick_vision_model().id == "v-shot"

    cfg.assist_screenshot_model_id = None
    assert kb_vision._pick_vision_model().id == "v-active"

    cfg.active_model_id = None
    cfg.active_model = None
    assert kb_vision._pick_vision_model().id == "v-first"

    # Verify think_params passed to chat_completion_create has thinking disabled
    cfg.assist_screenshot_model_id = "v-shot"
    png = tmp_path / "page.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    captured = {}

    def fake_create(model_cfg, messages, **kwargs):
        captured["model_id"] = model_cfg.id
        captured["think_params"] = kwargs.get("think_params")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])

    from services.llm import streaming
    monkeypatch.setattr(streaming, "chat_completion_create", fake_create)

    caption = kb_vision.vision_caption(png)
    assert caption == "summary"
    assert captured["model_id"] == "v-shot"
    assert captured["think_params"] == {"reasoning_effort": "none"}
