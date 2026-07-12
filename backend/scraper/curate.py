"""
Phase 3b: curate + convert the raw scrape into a clean JSON corpus for ingestion.

- Classifies each cached HTML page and each downloaded PDF as KEEP (student /
  institution-relevant) or DROP (staff profiles, taxonomy, news/events,
  procurement/tenders, vacancies, staff-admin) per the agreed filter.
- KEEP HTML pages -> boilerplate-stripped clean text -> one JSON doc each.
- KEEP PDFs -> table-aware text (framework DocumentLoader / pdfplumber) -> one
  JSON doc each.

Output JSON shape (matches services.document_loader.load_and_chunk_json):
    {"url", "title", "content", "category", "source_type", "scraped_at"}

Usage:
    python backend/scraper/curate.py [--src output/uom] [--dry-run]
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from bs4 import BeautifulSoup  # noqa: E402
from services.document_loader import DocumentLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("uom_curate")

# --- classification rules ----------------------------------------------------

# HTML URL paths to DROP (noise for a student assistant).
_HTML_DROP = re.compile(
    r"^/(staff\d*|staffs|people|taxonomy|node|user|"
    r"university_news|news|news_letter|announcements|"
    r"events|hot_events|hot-events|slideshow(-entry)?|"
    r"vacancy|vacancies)(/|$)",
    re.IGNORECASE,
)

# PDF path/filename substrings to DROP (procurement / tenders / vacancies /
# staff-admin — not student-relevant).
_PDF_DROP_SUBSTR = [
    "/procuments/", "/procurement", "/vacancy/", "/vacancies/",
    "dpc-", "dpc_", "_dpc", "ifb", "bid", "tender", "supplier-registration",
    "staff_attendance", "agrahara", "research-allowance", "research_allowance",
    "insurance", "specification_of_dpc", "extention_of_bid", "document-dpc",
]


def html_is_keep(path: str) -> bool:
    return not _HTML_DROP.match(path or "/")


def pdf_is_keep(rel_path: str) -> bool:
    low = rel_path.lower()
    return not any(s in low for s in _PDF_DROP_SUBSTR)


# --- HTML text extraction ----------------------------------------------------

_CANON = re.compile(r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"', re.I)
_STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form",
               "iframe", "svg", "button"]
_STRIP_SELECTORS = [
    ".region-header", ".region-footer", ".region-navigation", ".breadcrumb",
    "#navbar", "#footer", "#header", ".site-footer", ".site-header", ".menu",
    ".skip-link", ".sidebar", "#sidebar-first", "#sidebar-second",
    ".block-menu", ".social", ".search-block-form",
]
_MAIN_SELECTORS = ["main", "#content", ".region-content", "#main-content",
                   ".main-content", "#block-system-main", ".node__content"]


def extract_html(html: str):
    """Return (title, canonical_url, clean_text) from a cached page's HTML."""
    m = _CANON.search(html)
    url = m.group(1) if m else ""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = re.sub(r"\s*\|\s*University of Moratuwa.*$", "", soup.title.string).strip()

    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    for sel in _STRIP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    main = None
    for sel in _MAIN_SELECTORS:
        main = soup.select_one(sel)
        if main:
            break
    root = main or soup.body or soup

    if not title:
        h1 = root.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    text = root.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    clean = "\n".join(lines)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return title, url, clean


def _category_from_path(path: str) -> str:
    parts = [p for p in (path or "/").split("/") if p]
    return parts[0] if parts else "general"


def _slug(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Curate + convert the scrape to a JSON corpus")
    parser.add_argument("--src", default=str(Path(__file__).parent / "output" / "uom"))
    parser.add_argument("--min-chars", type=int, default=200,
                        help="skip pages/pdfs whose extracted text is shorter than this")
    parser.add_argument("--dry-run", action="store_true",
                        help="classify + count only, write nothing")
    parser.add_argument("--prune", action="store_true",
                        help="delete the raw source files classified as drop/empty "
                             "(keeps only the relevant documents on disk)")
    args = parser.parse_args()

    src = Path(args.src)
    html_raw = src / "html_raw"
    pdf_raw = src / "pdf_raw"
    out = src / "curated"
    out_html = out / "html"
    out_pdf = out / "pdf"
    if not args.dry_run:
        out_html.mkdir(parents=True, exist_ok=True)
        out_pdf.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    stats = {"html_keep": 0, "html_drop": 0, "html_empty": 0,
             "pdf_keep": 0, "pdf_drop": 0, "pdf_empty": 0, "pdf_error": 0}
    deleted = {"html": 0, "pdf": 0}
    unextractable_pdfs = []  # scanned/empty/error PDFs (reported before deletion)

    def _prune(path: Path, kind: str):
        if args.prune and not args.dry_run:
            try:
                path.unlink()
                deleted[kind] += 1
            except Exception as e:
                logger.warning(f"could not delete {path}: {e}")

    # --- HTML ---
    for f in sorted(html_raw.glob("*.html")):
        try:
            html = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        title, url, text = extract_html(html)
        path = urlparse(url).path if url else ""
        if not url or not html_is_keep(path):
            stats["html_drop"] += 1
            _prune(f, "html")
            continue
        if len(text) < args.min_chars:
            stats["html_empty"] += 1
            _prune(f, "html")
            continue
        stats["html_keep"] += 1
        if not args.dry_run:
            doc = {"url": url, "title": title or path,
                   "content": text, "category": _category_from_path(path),
                   "source_type": "html", "scraped_at": now}
            (out_html / f"{_slug(url)}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- PDF ---
    loader = DocumentLoader()
    pdfs = [p for p in pdf_raw.rglob("*.pdf")]
    for i, p in enumerate(pdfs, 1):
        rel = str(p.relative_to(pdf_raw)).replace("\\", "/")
        if not pdf_is_keep(rel):
            stats["pdf_drop"] += 1
            _prune(p, "pdf")
            continue
        try:
            text = loader.load_pdf(str(p))
        except Exception as e:
            stats["pdf_error"] += 1
            unextractable_pdfs.append({"file": rel, "reason": f"error: {e}"})
            logger.warning(f"PDF extract failed: {rel}: {e}")
            _prune(p, "pdf")
            continue
        if len(text.strip()) < args.min_chars:
            stats["pdf_empty"] += 1  # likely scanned/image-only
            unextractable_pdfs.append({"file": rel, "reason": "no text (scanned/image-only)"})
            _prune(p, "pdf")
            continue
        stats["pdf_keep"] += 1
        if not args.dry_run:
            # reconstruct a source URL from the mirrored path (uom.lk/...)
            url = "https://" + rel
            title = unquote(p.stem)
            parts = rel.split("/")
            category = parts[parts.index("files") + 1] if "files" in parts else "document"
            doc = {"url": url, "title": title, "content": text,
                   "category": category, "source_type": "pdf", "scraped_at": now}
            (out_pdf / f"{_slug(rel)}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        if i % 25 == 0:
            logger.info(f"  ...processed {i}/{len(pdfs)} PDFs")

    # Remove now-empty directories left behind by pruning.
    if args.prune and not args.dry_run:
        for d in sorted(pdf_raw.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except Exception:
                    pass

    logger.info("=" * 55)
    logger.info(f"HTML: kept {stats['html_keep']}, dropped {stats['html_drop']} (noise), "
               f"skipped {stats['html_empty']} (too short)")
    logger.info(f"PDF:  kept {stats['pdf_keep']}, dropped {stats['pdf_drop']} (procurement/vacancy/admin), "
               f"skipped {stats['pdf_empty']} (empty/scanned), errors {stats['pdf_error']}")
    if args.prune and not args.dry_run:
        logger.info(f"PRUNED (deleted) {deleted['html']} HTML + {deleted['pdf']} PDF source files")
    if not args.dry_run:
        logger.info(f"Curated corpus written to {out}")
    logger.info("=" * 55)
    stats["deleted"] = deleted
    (src / "curation_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (src / "unextractable_pdfs.json").write_text(
        json.dumps(unextractable_pdfs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
