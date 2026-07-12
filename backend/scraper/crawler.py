"""
BFS crawler over www.uom.lk — discovers HTML pages and PDF links.

Respects robots.txt for HTML page crawling (skips disallowed paths like
/admin/, /user/login/, /search/). PDF links are recorded regardless of their
path's robots.txt status — the caller decides whether to fetch them (see
pdf_fetcher.py) since a direct file download is not the same as crawling.

Only follows links within ALLOWED_DOMAINS. Anything else (lms.uom.lk,
codl.lk, external sites, etc.) is recorded as "discovered but not crawled"
in the manifest rather than silently followed or silently dropped.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
from urllib.parse import urljoin, urlparse, urldefrag
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("uom_scraper.crawler")

USER_AGENT = "UOM-KB-Ingest-Bot/1.0 (contact: ruzainiahmedh0706@gmail.com)"
SEED_URLS = ["https://www.uom.lk/"]
ALLOWED_DOMAINS = {"uom.lk", "www.uom.lk"}
# Gentle by default: a sustained 0.6s-delay crawl tripped uom.lk's rate limiter
# and got our IP temporarily banned. 2s single-threaded (~0.5 req/s) is far
# safer for a small institutional server. Also back off hard on 429/503.
REQUEST_DELAY_SEC = 2.0
REQUEST_TIMEOUT_SEC = 25
BACKOFF_ON_THROTTLE_SEC = 60
# High ceiling so the crawl runs to exhaustion on the real site (~several
# thousand pages incl. every staff/course/notice page); this is only a safety
# valve against pagination/taxonomy loops, not an intended stopping point.
MAX_PAGES = 15000

# Pagination query strings (?page=N) re-list content that is already reachable
# via direct links and balloon the queue (they were the main driver of the
# runaway crawl that got us rate-limited). Skip them.
SKIP_URL_PATTERNS = [
    re.compile(r"[?&]page=", re.IGNORECASE),
]

# Extensions that are clearly not HTML pages worth crawling into (but may still
# be recorded, e.g. PDFs are handled specially below).
NON_HTML_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".zip", ".rar",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}

TRACKING_PARAM_RE = re.compile(r"^(utm_|fbclid|gclid|sessionid)", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Strip fragment + trailing slash + tracking query params for dedup."""
    url, _frag = urldefrag(url)
    parsed = urlparse(url)
    if parsed.query:
        kept = [
            kv for kv in parsed.query.split("&")
            if kv and not TRACKING_PARAM_RE.match(kv.split("=")[0])
        ]
        query = "&".join(kept)
    else:
        query = ""
    path = parsed.path.rstrip("/") or "/"
    normalized = parsed._replace(path=path, query=query, fragment="").geturl()
    return normalized


def domain_of(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


@dataclass
class CrawlResult:
    html_pages: dict = field(default_factory=dict)   # url -> {"title": str, "file": str|None}
    pdf_urls: dict = field(default_factory=dict)      # pdf_url -> {"found_on": [urls]}
    skipped_robots: list = field(default_factory=list)
    skipped_domain: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    truncated: bool = False


class RobotsCache:
    def __init__(self, session: requests.Session):
        self._session = session
        self._parsers: dict = {}

    def _get(self, domain: str) -> Optional[robotparser.RobotFileParser]:
        if domain in self._parsers:
            return self._parsers[domain]
        rp = robotparser.RobotFileParser()
        robots_url = f"https://{domain}/robots.txt"
        try:
            resp = self._session.get(robots_url, timeout=REQUEST_TIMEOUT_SEC)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.parse([])  # no robots.txt -> allow everything
        except requests.RequestException as e:
            logger.warning(f"Could not fetch robots.txt for {domain}: {e}")
            rp.parse([])
        self._parsers[domain] = rp
        return rp

    def can_fetch(self, url: str) -> bool:
        domain = domain_of(url)
        rp = self._get(domain)
        if rp is None:
            return True
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True


def _should_skip_url(url: str) -> bool:
    return any(p.search(url) for p in SKIP_URL_PATTERNS)


def crawl(seed_urls=None, max_pages: int = MAX_PAGES,
          on_progress=None, page_sink=None, pdf_sink=None,
          cache_lookup=None) -> CrawlResult:
    """BFS-crawl the allowed domains starting from seed_urls.

    on_progress(visited_count, queued_count, current_url): progress logging.

    page_sink(url, html, title) -> filename: stream each fetched HTML page to
    disk immediately (bounded memory, crash-safe).

    pdf_sink(pdf_url, found_on_url): called the first time each PDF URL is
    discovered, so the caller can download it INCREMENTALLY (an interruption
    then never loses already-downloaded PDFs).

    cache_lookup(url) -> html|None: RESUME support. If it returns cached HTML
    for a URL, that page is parsed from cache WITHOUT any network request
    (no delay, no server load) — so re-running after an interruption re-reads
    already-crawled pages instantly and only hits the site for genuinely new
    URLs.
    """
    seed_urls = seed_urls or SEED_URLS
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    robots = RobotsCache(session)

    result = CrawlResult()
    visited = set()
    queue = deque(normalize_url(u) for u in seed_urls)
    queued_set = set(queue)

    def _record_pdf(pdf_url, source_url):
        first_time = pdf_url not in result.pdf_urls
        entry = result.pdf_urls.setdefault(pdf_url, {"found_on": []})
        if source_url not in entry["found_on"]:
            entry["found_on"].append(source_url)
        if first_time and pdf_sink is not None:
            try:
                pdf_sink(pdf_url, source_url)
            except Exception as e:
                result.errors.append({"url": pdf_url, "error": f"pdf_sink failed: {e}"})

    while queue:
        if len(visited) >= max_pages:
            result.truncated = True
            logger.warning(f"Hit MAX_PAGES={max_pages}, stopping crawl (queue had "
                           f"{len(queue)} more URLs).")
            break

        url = queue.popleft()
        queued_set.discard(url)
        if url in visited:
            continue
        visited.add(url)

        if domain_of(url) not in ALLOWED_DOMAINS:
            result.skipped_domain.append(url)
            continue

        if _should_skip_url(url):
            continue

        if not robots.can_fetch(url):
            result.skipped_robots.append(url)
            continue

        # Resume: use cached HTML if we already crawled this page (no network).
        html = cache_lookup(url) if cache_lookup is not None else None
        from_cache = html is not None

        if not from_cache:
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT_SEC)
            except requests.RequestException as e:
                result.errors.append({"url": url, "error": str(e)})
                continue

            # Back off hard if the server signals throttling, then retry once.
            if resp.status_code in (429, 503):
                logger.warning(f"HTTP {resp.status_code} (throttled) on {url}; "
                               f"backing off {BACKOFF_ON_THROTTLE_SEC}s")
                time.sleep(BACKOFF_ON_THROTTLE_SEC)
                try:
                    resp = session.get(url, timeout=REQUEST_TIMEOUT_SEC)
                except requests.RequestException as e:
                    result.errors.append({"url": url, "error": str(e)})
                    continue

            time.sleep(REQUEST_DELAY_SEC)

            if resp.status_code != 200:
                result.errors.append({"url": url, "error": f"HTTP {resp.status_code}"})
                continue

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue  # not a page we can parse for links
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        filename = None
        if page_sink is not None and not from_cache:
            try:
                filename = page_sink(url, html, title)
            except Exception as e:
                result.errors.append({"url": url, "error": f"page_sink failed: {e}"})
        result.html_pages[url] = {"title": title, "file": filename, "cached": from_cache}

        if on_progress:
            on_progress(len(visited), len(queue), url)

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            absolute = normalize_url(urljoin(url, href))

            if is_pdf_url(absolute):
                # Only download PDFs hosted on the university's own domain.
                # PDFs linked to other hosts (lms.uom.lk LMS, external
                # publishers like intechopen/edas, etc.) are out of scope —
                # record them as skipped rather than downloading.
                if domain_of(absolute) in ALLOWED_DOMAINS:
                    _record_pdf(absolute, url)
                elif absolute not in result.skipped_domain:
                    result.skipped_domain.append(absolute)
                continue

            path_ext = "." + absolute.rsplit(".", 1)[-1].lower() if "." in absolute.rsplit("/", 1)[-1] else ""
            if path_ext in NON_HTML_EXTENSIONS:
                continue

            if domain_of(absolute) not in ALLOWED_DOMAINS:
                if absolute not in result.skipped_domain:
                    result.skipped_domain.append(absolute)
                continue

            if _should_skip_url(absolute):
                continue

            if absolute not in visited and absolute not in queued_set:
                queue.append(absolute)
                queued_set.add(absolute)

    return result
