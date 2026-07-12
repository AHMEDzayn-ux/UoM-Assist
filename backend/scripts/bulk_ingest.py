"""
Bulk-ingest the curated uom.lk corpus into a client's knowledge base.

Creates (or resets) the client, then indexes every curated JSON document
(HTML pages + PDFs) in ONE index_documents call so the BM25 index and FAISS
store are built once over the whole corpus (calling it per-batch would rebuild
BM25 from only the last batch — see rag_pipeline.index_documents).

Usage (from backend/, venv active):
    python scripts/bulk_ingest.py                 # create uom client + ingest
    python scripts/bulk_ingest.py --reset         # wipe existing uom collection first
    python scripts/bulk_ingest.py --slug uom --src scraper/output/uom/curated
"""

import os  # OpenMP guard must precede torch/faiss (see main.py)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sentence_transformers  # noqa: F401  (import before faiss)

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from database import init_db, SessionLocal
from services import client_store, auth_service
from services.rag_pipeline import MultiClientRAGPipeline
from logger import get_logger

logger = get_logger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Bulk-ingest curated corpus into a client KB")
    ap.add_argument("--slug", default="uom")
    ap.add_argument("--name", default="University of Moratuwa")
    ap.add_argument("--domain", default="university")
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1] /
                                         "scraper" / "output" / "uom" / "curated"))
    ap.add_argument("--reset", action="store_true",
                    help="delete any existing collection for this slug first")
    args = ap.parse_args()

    src = Path(args.src)
    docs = []
    for sub in ("html", "pdf"):
        for jf in sorted((src / sub).glob("*.json")):
            try:
                docs.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"skip {jf.name}: {e}")
    if not docs:
        logger.error(f"No curated docs found under {src}")
        sys.exit(1)
    print(f"[ingest] loaded {len(docs)} curated documents from {src}", flush=True)

    init_db()
    db = SessionLocal()
    manager = MultiClientRAGPipeline()
    try:
        # Owner = bootstrap superadmin (so the client shows in /admin).
        auth_service.bootstrap_admin(db)
        admin = auth_service.get_user_by_email(db, __import__("config").get_settings().admin_email)
        owner_id = admin.id if admin else None

        existing = client_store.get_client(db, args.slug)
        if existing and args.reset:
            logger.info(f"--reset: deleting existing collection for '{args.slug}'")
            manager.delete_pipeline(args.slug)
            client_store.delete_client(db, args.slug)
            existing = None

        if not existing:
            client = client_store.create_client(
                db, slug=args.slug, name=args.name,
                description="University of Moratuwa student & staff assistant",
                domain=args.domain, owner_id=owner_id,
            )
            logger.info(f"Created client '{client.slug}' (domain={client.domain})")
        else:
            client = existing
            logger.info(f"Using existing client '{client.slug}' (domain={client.domain})")

        print("[ingest] building pipeline (loads embedding model)...", flush=True)
        pipeline = manager.create_pipeline(
            client_id=client.slug,
            system_role=client_store.resolve_persona(client),
            domain=client.domain,
        )

        # One combined array file -> single index_documents call (correct BM25).
        tmp = Path(tempfile.mkdtemp()) / "uom_corpus.json"
        tmp.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")

        print(f"[ingest] indexing {len(docs)} docs -> embedding all chunks (minutes)...", flush=True)
        try:
            result = pipeline.index_documents(
                file_paths=[str(tmp)],
                metadata={"client": args.slug, "source": "uom.lk"},
            )
        except Exception:
            import traceback
            print("[ingest] index_documents FAILED:", flush=True)
            traceback.print_exc()
            raise
        chunks = result.get("chunks_created", result.get("total_chunks", 0)) if isinstance(result, dict) else 0

        # Record document rows so the admin UI shows a real KB count.
        for d in docs:
            try:
                client_store.add_document(
                    db, client_slug=args.slug,
                    filename=d.get("title") or d.get("url") or "document",
                    doc_type=d.get("source_type", "doc"), chunk_count=0,
                )
            except Exception:
                pass

        logger.info("=" * 55)
        logger.info(f"DONE. Ingested {len(docs)} documents -> {chunks} chunks "
                   f"into collection client_{args.slug}")
        logger.info("=" * 55)
        try:
            tmp.unlink(); tmp.parent.rmdir()
        except OSError:
            pass
    finally:
        db.close()


if __name__ == "__main__":
    main()
