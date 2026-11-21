import base64
import os
import platform
import subprocess
import tempfile

MAX_RESUME_PDF_PAGES = 20
MAX_DOCX_ZIP_ENTRIES = 1000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_DOCX_XML_BYTES = 20 * 1024 * 1024


def _pdf_to_images_base64(
    file_path: str,
    dpi: int = 150,
    *,
    max_pages: int | None = None,
) -> list[str]:
    """用 PyMuPDF 将 PDF 每页转为 PNG 的 base64，无系统依赖。"""
    try:
        import fitz
    except ImportError:
        raise ValueError("解析图片版 PDF 需要安装 PyMuPDF：pip install pymupdf") from None
    limit = MAX_RESUME_PDF_PAGES if max_pages is None else max(1, int(max_pages))
    out = []
    doc = fitz.open(file_path)
    try:
        if len(doc) > limit:
            raise ValueError(f"PDF 页数过多（{len(doc)} 页，上限 {limit} 页）")
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            png_bytes = pix.tobytes("png")
            out.append(base64.b64encode(png_bytes).decode("ascii"))
    finally:
        doc.close()
    return out


def parse_pdf(file_path: str) -> str:
    """PDF 简历统一由 VL 模型解析（需在设置中配置支持识图的模型）。"""
    from services.llm import has_vision_model, vision_extract_text
    if not has_vision_model():
        raise ValueError(
            "上传 PDF 需要先配置支持识图的模型。请打开「设置」→ 选择一个带「识图」的模型并填写 API Key → 保存后再上传 PDF；"
            "或改为上传 DOCX / TXT 格式的简历。"
        )
    images_b64 = _pdf_to_images_base64(file_path)
    if not images_b64:
        raise ValueError("PDF 无有效页面")
    return vision_extract_text(images_b64)


def _validate_docx_zip(z) -> None:
    infos = z.infolist()
    if len(infos) > MAX_DOCX_ZIP_ENTRIES:
        raise ValueError(f"DOCX 压缩包条目数超过上限（{len(infos)} > {MAX_DOCX_ZIP_ENTRIES}）")
    total_uncompressed = 0
    for info in infos:
        if info.file_size < 0:
            raise ValueError("无效的 DOCX 文件")
        total_uncompressed += info.file_size
        if total_uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise ValueError("DOCX 解压后体积超过安全上限，请检查文件是否异常")


def _parse_docx(content: bytes) -> str:
    """Extract text from a .docx file with Zip-bomb size and entry guards."""
    import zipfile
    import xml.etree.ElementTree as ET
    from io import BytesIO

    paragraphs = []
    try:
        with zipfile.ZipFile(BytesIO(content)) as z:
            _validate_docx_zip(z)
            if "word/document.xml" not in z.namelist():
                raise ValueError("无效的 DOCX 文件")
            with z.open("word/document.xml") as f:
                xml_bytes = f.read(MAX_DOCX_XML_BYTES + 1)
            if len(xml_bytes) > MAX_DOCX_XML_BYTES:
                raise ValueError("DOCX 正文 XML 解压后体积超过安全上限")
            try:
                root = ET.fromstring(xml_bytes)
            except ET.ParseError:
                raise ValueError("无效的 DOCX 文件") from None
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for p in root.iter(f"{{{ns['w']}}}p"):
                texts = [t.text for t in p.iter(f"{{{ns['w']}}}t") if t.text]
                if texts:
                    paragraphs.append("".join(texts))
    except zipfile.BadZipFile:
        raise ValueError("无效的 DOCX 文件") from None
    return "\n".join(paragraphs)


def _parse_doc(content: bytes) -> str:
    """Extract text from legacy .doc (Word 97-2003). Uses system tool: macOS textutil, Linux antiword/catdoc."""
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        if platform.system() == "Darwin":
            out_path = tmp_path + ".txt"
            try:
                subprocess.run(
                    ["textutil", "-convert", "txt", tmp_path, "-output", out_path],
                    check=True, capture_output=True, timeout=30
                )
                with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            finally:
                if os.path.exists(out_path):
                    os.unlink(out_path)
        for cmd in (["antiword", "-t", tmp_path], ["catdoc", "-w", tmp_path]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
                if r.returncode == 0 and (r.stdout or "").strip():
                    return r.stdout.strip()
            except FileNotFoundError:
                continue
        raise ValueError(
            "当前环境无法解析 .doc 文件。macOS 无需额外安装；"
            "Linux 请安装 antiword 或 catdoc（如 apt install antiword），或将文件另存为 .docx 后上传。"
        )
    finally:
        os.unlink(tmp_path)


def parse_resume_bytes(content: bytes, filename: str) -> str:
    """Parse resume from uploaded bytes. Supports PDF, DOCX, DOC, TXT, MD."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return parse_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)
    elif ext == ".docx":
        return _parse_docx(content)
    elif ext == ".doc":
        return _parse_doc(content)
    elif ext in (".txt", ".md"):
        return content.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"不支持的文件格式: {ext}，请上传 PDF / DOCX / TXT / MD 文件")


def summarize_resume(text: str) -> str:
    """Return the resume text as-is (LLM will handle interpretation)."""
    if len(text) > 8000:
        return text[:8000] + "\n\n[简历内容过长，已截取前 8000 字符]"
    return text
