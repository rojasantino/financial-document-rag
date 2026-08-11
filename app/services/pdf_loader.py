"""
Step 1: PDF -> plain text, one entry per page.
Preserving the page number is what lets us cite sources later
("Annual_Report.pdf, Page 42").
"""
from typing import List, Dict
import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """Returns a list of {"page": int, "text": str} for every page with content."""
    document = fitz.open(file_path)
    pages = []

    for page_number, page in enumerate(document):
        text = page.get_text()
        if text.strip():
            pages.append({"page": page_number + 1, "text": text})

    document.close()
    return pages


def is_scanned_pdf(file_path: str, min_chars_per_page: int = 20) -> bool:
    """
    Heuristic: if PyMuPDF extracts almost no text, the PDF is probably
    a scanned image and needs OCR (see README "Extending the project").
    """
    pages = extract_text_from_pdf(file_path)
    if not pages:
        return True
    avg_chars = sum(len(p["text"]) for p in pages) / len(pages)
    return avg_chars < min_chars_per_page
