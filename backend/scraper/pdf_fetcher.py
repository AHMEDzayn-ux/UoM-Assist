"""
Downloads PDF files discovered by crawler.py.

Per an explicit decision made with the project owner: PDFs are fetched
directly regardless of robots.txt path restrictions on the page that hosts
them (e.g. uom.lk disallows crawling /sites/default/files/, which is where
many of its PDFs actually live) — a direct file download is being treated
here as a one-off document fetch, not page crawling/indexing. HTML page
crawling in crawler.py still fully respects robots.txt.
"""

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

logger = logging.getLogger("uom_scraper.pdf_fetcher")

USER_AGENT = "UOM-KB-Ingest-Bot/1.0 (contact: ruzainiahmedh0706@gmail.com)"
REQUEST_DELAY_SEC = 0.5
REQUEST_TIMEOUT_SEC = 45


def _sanitize(segment: str) -> str:
    """Make a single path segment filesystem-safe and length-bounded."""
    segment = segment.strip()
    segment = "".join(c if c.isalnum() or c in "-_." else "_" for c in segment)
    segment = segment.strip("._") or "_"
    return segment[:120]


def _dest_for(url: str, out_dir: Path) -> tuple:
    """Build a hierarchical destination mirroring the PDF's URL path, e.g.
    https://uom.lk/sites/default/files/notices/files/Aptitude Test.pdf
      -> <out_dir>/uom.lk/sites/default/files/notices/files/Aptitude_Test.pdf

    Returns (relative_path_str, absolute_dest_path).
    """
    parsed = urlparse(url)
    host = (parsed.netloc or "unknown").lower()
    if host.startswith("www."):
        host = host[4:]  # fold www.uom.lk and uom.lk into one tree

    parts = [p for p in unquote(parsed.path).split("/") if p] or ["file.pdf"]
    *dir_parts, filename = parts

    filename = _sanitize(filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    rel = Path(_sanitize(host))
    for d in dir_parts:
        rel = rel / _sanitize(d)
    rel = rel / filename
    return str(rel).replace("\\", "/"), (out_dir / rel)


def download_one(url: str, found_on, out_dir: Path,
                 session: requests.Session, delay: float = REQUEST_DELAY_SEC) -> dict:
    """Download a single PDF into the URL-mirroring hierarchy under out_dir.

    Resume-safe: if the destination already exists (from a prior run), it is
    NOT re-downloaded — the existing file's size is reported with
    status "skipped". Sleeps `delay` after any network fetch.

    Returns a manifest entry:
    {"url", "found_on", "status": "ok"|"skipped"|"error", "file", "bytes", "error"?}
    """
    entry = {"url": url, "found_on": list(found_on or [])}
    rel_path, dest = _dest_for(url, out_dir)

    # Resume: already have it on disk -> skip the network entirely.
    if dest.exists():
        entry["status"] = "skipped"
        entry["file"] = rel_path
        entry["bytes"] = dest.stat().st_size
        return entry

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT_SEC, stream=True)
        if resp.status_code != 200:
            entry["status"] = "error"
            entry["error"] = f"HTTP {resp.status_code}"
            time.sleep(delay)
            return entry

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Disambiguate the rare case where two distinct URLs sanitize to the
        # same destination path (append a short URL hash before .pdf).
        if dest.exists():
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
            dest = dest.with_name(f"{dest.stem}__{digest}.pdf")
            rel_path = str(dest.relative_to(out_dir)).replace("\\", "/")

        size = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)

        entry["status"] = "ok"
        entry["file"] = rel_path
        entry["bytes"] = size
        time.sleep(delay)
    except requests.RequestException as e:
        entry["status"] = "error"
        entry["error"] = str(e)
        time.sleep(delay)
    return entry


def download_pdfs(pdf_urls: dict, out_dir: Path, on_progress=None) -> list:
    """Batch-download every PDF URL (used for a standalone download pass).
    Prefer the incremental pdf_sink path in run_scrape.py for live crawls.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    entries = []
    total = len(pdf_urls)
    for i, (url, meta) in enumerate(pdf_urls.items(), start=1):
        entry = download_one(url, meta.get("found_on", []), out_dir, session)
        entries.append(entry)
        if on_progress:
            on_progress(i, total, url, entry["status"])
    return entries
