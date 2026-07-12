"""
Phase 3a CLI: crawl www.uom.lk, discover every PDF link, and download the raw
PDF files INCREMENTALLY (as each is discovered). Does NOT convert anything to
JSON yet (that's Phase 3b, gated on explicit confirmation after reviewing this
phase's manifest).

Resumable + gentle:
- Each crawled page's HTML is streamed to output/uom/html_raw/ immediately.
- Re-running RESUMES: pages already cached in html_raw/ are re-read from disk
  (no network, no server load) and PDFs already on disk are skipped — so an
  interruption (or an IP rate-limit ban) never loses progress, and resuming
  only hits the site for genuinely new URLs.
- PDFs download as they are discovered, so an interruption keeps everything
  fetched so far.
- Polite crawl rate (see crawler.REQUEST_DELAY_SEC) with 429/503 backoff.

Usage:
    python backend/scraper/run_scrape.py [--max-pages N] [--out DIR]
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

import requests  # noqa: E402
from scraper.crawler import crawl, MAX_PAGES, SEED_URLS, USER_AGENT, REQUEST_DELAY_SEC  # noqa: E402
from scraper.pdf_fetcher import download_one  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("uom_scraper.run")


def _url_to_filename(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.html"


def main():
    parser = argparse.ArgumentParser(description="Crawl uom.lk and download every PDF (Phase 3a)")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    parser.add_argument("--out", type=str, default=str(Path(__file__).parent / "output" / "uom"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    html_raw_dir = out_dir / "html_raw"
    pdf_raw_dir = out_dir / "pdf_raw"
    html_raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_raw_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    cached_already = len(list(html_raw_dir.glob("*.html")))
    logger.info(f"Starting crawl from {SEED_URLS}, max_pages={args.max_pages}, "
               f"delay={REQUEST_DELAY_SEC}s, {cached_already} pages already cached (resume)")

    pdf_session = requests.Session()
    pdf_session.headers.update({"User-Agent": USER_AGENT})
    pdf_results = []          # running list of pdf download manifest entries
    pdf_counters = {"ok": 0, "skipped": 0, "error": 0}

    manifest_path = out_dir / "manifest.json"

    # --- resume: read already-crawled pages from the HTML cache (no network) --
    def cache_lookup(url):
        fpath = html_raw_dir / _url_to_filename(url)
        if fpath.exists():
            try:
                return fpath.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    # --- stream each newly fetched page to disk ------------------------------
    def page_sink(url, html, title):
        filename = _url_to_filename(url)
        (html_raw_dir / filename).write_text(html, encoding="utf-8")
        return filename

    # --- download each PDF the moment it is discovered -----------------------
    def pdf_sink(pdf_url, found_on):
        entry = download_one(pdf_url, [found_on], pdf_raw_dir, pdf_session,
                             delay=REQUEST_DELAY_SEC)
        pdf_results.append(entry)
        pdf_counters[entry["status"]] = pdf_counters.get(entry["status"], 0) + 1
        n = len(pdf_results)
        if entry["status"] == "error" or n % 5 == 0:
            logger.info(f"[pdf {n}] {entry['status']}: {pdf_url}")
        return

    def on_page(visited_count, queued_count, current_url):
        if visited_count % 25 == 0 or visited_count <= 5:
            logger.info(f"[crawl] page {visited_count} (queue={queued_count}, "
                       f"pdfs={len(pdf_results)}): {current_url}")

    result = crawl(
        max_pages=args.max_pages,
        on_progress=on_page,
        page_sink=page_sink,
        pdf_sink=pdf_sink,
        cache_lookup=cache_lookup,
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    ok = pdf_counters.get("ok", 0)
    skipped = pdf_counters.get("skipped", 0)
    failed = pdf_counters.get("error", 0)
    total_bytes = sum(e.get("bytes", 0) for e in pdf_results if e["status"] in ("ok", "skipped"))

    html_manifest = {
        url: {"file": page["file"], "title": page["title"]}
        for url, page in result.html_pages.items()
    }

    manifest = {
        "phase": "3a-pdf-download",
        "started_at": started_at,
        "finished_at": finished_at,
        "seed_urls": SEED_URLS,
        "max_pages": args.max_pages,
        "request_delay_sec": REQUEST_DELAY_SEC,
        "pages_crawled": len(result.html_pages),
        "pages_truncated": result.truncated,
        "pdf_links_discovered": len(result.pdf_urls),
        "pdf_downloaded_ok": ok,
        "pdf_skipped_existing": skipped,
        "pdf_downloaded_failed": failed,
        "pdf_total_bytes": total_bytes,
        "skipped_robots_count": len(result.skipped_robots),
        "skipped_domain_count": len(set(result.skipped_domain)),
        "errors_count": len(result.errors),
        "html_pages": html_manifest,
        "pdf_downloads": pdf_results,
        "skipped_robots": result.skipped_robots,
        "skipped_domain": sorted(set(result.skipped_domain)),
        "errors": result.errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info(f"DONE. Pages crawled/read: {len(result.html_pages)} (truncated={result.truncated})")
    logger.info(f"PDFs discovered: {len(result.pdf_urls)} -> downloaded: {ok}, "
               f"already-had: {skipped}, failed: {failed}, "
               f"total size: {total_bytes / 1e6:.1f} MB")
    logger.info(f"Manifest written to {manifest_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
