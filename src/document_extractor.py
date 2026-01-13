"""Document text extraction for various file formats."""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def extract_text_from_file(file_path: str | Path, file_content: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Extract plain text content from a document file.
    
    Args:
        file_path: Path to the file (used to determine format)
        file_content: Optional file content bytes. If not provided, reads from file_path.
    
    Returns:
        Tuple of (extracted_text, format_name)
    
    Raises:
        ValueError: If the file format is not supported
        Exception: If extraction fails
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if file_content is None:
        file_content = path.read_bytes()
    
    extractors = {
        '.txt': ('Plain Text', _extract_txt),
        '.md': ('Markdown', _extract_markdown),
        '.pdf': ('PDF', _extract_pdf),
        '.docx': ('Word Document', _extract_docx),
        '.doc': ('Word Document (Legacy)', _extract_doc),
        '.rtf': ('Rich Text Format', _extract_rtf),
        '.epub': ('EPUB', _extract_epub),
        '.odt': ('OpenDocument Text', _extract_odt),
        '.html': ('HTML', _extract_html),
        '.htm': ('HTML', _extract_html),
    }
    
    if suffix not in extractors:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    format_name, extractor = extractors[suffix]
    
    try:
        text = extractor(file_content)
        text = _clean_extracted_text(text)
        return text, format_name
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", format_name, e)
        raise


def _clean_extracted_text(text: str) -> str:
    """Clean up extracted text by normalizing whitespace and removing artifacts."""
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Remove leading/trailing whitespace from each line while preserving structure
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Remove leading/trailing whitespace from entire text
    text = text.strip()
    
    return text


def _extract_txt(content: bytes) -> str:
    """Extract text from plain text file."""
    # Try common encodings
    for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Fallback with error handling
    return content.decode('utf-8', errors='replace')


def _extract_markdown(content: bytes) -> str:
    """Extract text from Markdown file (returns as-is, it's already text)."""
    text = _extract_txt(content)
    
    # Optionally strip markdown formatting to get plain text
    # For now, keep markdown as-is since speaker tags might be in it
    return text


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF extraction. Install with: pip install pypdf")
    
    reader = PdfReader(io.BytesIO(content))
    text_parts = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    
    return '\n\n'.join(text_parts)


def _extract_docx(content: bytes) -> str:
    """Extract text from Word .docx file."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for Word document extraction. Install with: pip install python-docx")
    
    doc = Document(io.BytesIO(content))
    text_parts = []
    
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    
    return '\n\n'.join(text_parts)


def _extract_doc(content: bytes) -> str:
    """Extract text from legacy Word .doc file."""
    # Legacy .doc files are harder to parse without external tools
    # Try to use antiword or similar if available, otherwise raise helpful error
    try:
        # Try using python-docx2txt which can handle some .doc files
        import docx2txt
        return docx2txt.process(io.BytesIO(content))
    except ImportError:
        pass
    
    # Fallback: try to extract any readable text
    try:
        text = content.decode('utf-8', errors='ignore')
        # Filter to printable characters
        text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
        if len(text) > 100:  # Seems like we got something
            return text
    except Exception:
        pass
    
    raise ValueError(
        "Legacy .doc files require additional tools. "
        "Please convert to .docx format or save as .txt/.pdf."
    )


def _extract_rtf(content: bytes) -> str:
    """Extract text from RTF file."""
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        raise ImportError("striprtf is required for RTF extraction. Install with: pip install striprtf")
    
    rtf_content = content.decode('utf-8', errors='replace')
    return rtf_to_text(rtf_content)


def _extract_epub(content: bytes) -> str:
    """Extract text from EPUB file."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("ebooklib and beautifulsoup4 are required for EPUB extraction.")
    
    book = epub.read_epub(io.BytesIO(content))
    text_parts = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n')
            if text.strip():
                text_parts.append(text.strip())
    
    return '\n\n'.join(text_parts)


def _extract_odt(content: bytes) -> str:
    """Extract text from OpenDocument Text file."""
    try:
        from odf import text as odf_text
        from odf.opendocument import load
    except ImportError:
        raise ImportError("odfpy is required for ODT extraction. Install with: pip install odfpy")
    
    doc = load(io.BytesIO(content))
    text_parts = []
    
    for para in doc.getElementsByType(odf_text.P):
        para_text = ''.join(node.data for node in para.childNodes if hasattr(node, 'data'))
        if para_text.strip():
            text_parts.append(para_text)
    
    return '\n\n'.join(text_parts)


def _extract_html(content: bytes) -> str:
    """Extract text from HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required for HTML extraction.")
    
    # Detect encoding
    html_text = None
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            html_text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if html_text is None:
        html_text = content.decode('utf-8', errors='replace')
    
    soup = BeautifulSoup(html_text, 'html.parser')
    
    # Remove script, style, nav, header, footer elements
    for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
        element.decompose()
    
    # Try to find main content
    main_content = soup.find('main') or soup.find('article') or soup.find('body') or soup
    
    text = main_content.get_text(separator='\n')
    return text


def get_supported_formats() -> list[dict]:
    """Return list of supported file formats with descriptions."""
    return [
        {"extension": ".txt", "name": "Plain Text", "mime": "text/plain"},
        {"extension": ".md", "name": "Markdown", "mime": "text/markdown"},
        {"extension": ".pdf", "name": "PDF Document", "mime": "application/pdf"},
        {"extension": ".docx", "name": "Word Document", "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        {"extension": ".doc", "name": "Word Document (Legacy)", "mime": "application/msword"},
        {"extension": ".rtf", "name": "Rich Text Format", "mime": "application/rtf"},
        {"extension": ".epub", "name": "EPUB eBook", "mime": "application/epub+zip"},
        {"extension": ".odt", "name": "OpenDocument Text", "mime": "application/vnd.oasis.opendocument.text"},
        {"extension": ".html", "name": "HTML", "mime": "text/html"},
        {"extension": ".htm", "name": "HTML", "mime": "text/html"},
    ]
