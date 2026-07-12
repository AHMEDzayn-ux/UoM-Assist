"""
FastAPI Application for Multi-Tenant RAG Chatbot System
Phase 7: REST API Implementation with Security
"""

# IMPORTANT: faiss ↔ PyTorch OpenMP conflict fix (Windows).
# The multilingual embedding + reranker models load a sentencepiece/XLM-R stack
# whose OpenMP runtime segfaults against faiss's unless (a) OpenMP is told to
# tolerate duplicate runtimes and use a single thread, and (b) torch is imported
# before faiss. Both must happen before ANY other import. Do not reorder.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sentence_transformers  # noqa: F401  (side-effect: init torch/OpenMP first)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from logger import get_logger
from api.clients import router as clients_router
from api.documents import router as documents_router
from api.query import router as query_router
from api.voice import router as voice_router
from api.public import router as public_router
from api.meta import router as meta_router
from api.auth_routes import router as auth_router
from integrations.whatsapp_bot import router as whatsapp_router
from config import settings
from security import SecurityMiddleware
from database import init_db, SessionLocal
from services.client_store import reconcile_disk_collections

logger = get_logger(__name__)


def _seed_uom(db):
    """Guarantee the shipped 'uom' knowledge base presents as a University of
    Moratuwa (domain=university) client with a portal client-admin.

    The ~25MB uom FAISS index ships inside the image, so on a fresh container
    reconcile_disk_collections() imports it — but as the *generic* domain. This
    idempotent seed promotes it to the university template (correct persona,
    bot name, greeting, and university action tools) and ensures a client-admin
    exists (credentials from env, so nothing sensitive lives in git). Once the
    domain is 'university' the promotion block is skipped, preserving any later
    edits an operator makes in the Admin Console.
    """
    import os
    from services.client_store import get_client, update_client
    from services.auth_service import get_user_by_email, create_user
    from domain_templates import get_template

    uom = get_client(db, "uom")
    if uom is None:
        return  # index not present in this build; nothing to seed
    if uom.domain != "university":
        tpl = get_template("university")
        update_client(db, "uom", domain="university", persona=None,
                      name="University of Moratuwa",
                      bot_name=tpl.bot_name, greeting=tpl.greeting)
        logger.info("Seeded 'uom' client as domain=university")

    admin_email = os.getenv("UOM_ADMIN_EMAIL", "uom-admin@uom.lk")
    if not get_user_by_email(db, admin_email):
        create_user(db, email=admin_email,
                    password=os.getenv("UOM_ADMIN_PASSWORD", "UoM@Admin2026"),
                    name="UoM Portal Admin", role="client_admin",
                    client_slug="uom")
        logger.info(f"Created uom client-admin: {admin_email}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure DB tables exist and import pre-existing on-disk clients."""
    init_db()
    db = SessionLocal()
    try:
        imported = reconcile_disk_collections(db)
        if imported:
            logger.info(f"Reconciled {imported} on-disk client collection(s) into DB")
        # Seed the first operator account (claims any legacy clients).
        from services.auth_service import bootstrap_admin
        bootstrap_admin(db)
        _seed_uom(db)
    except Exception as e:
        logger.error(f"Startup reconciliation failed: {e}")
    finally:
        db.close()
    yield

# Log configuration at startup
logger.info(f"Starting RAG API...")
logger.info(f"Environment: {settings.environment}")
logger.info(f"Groq API Key configured: {bool(settings.groq_api_key)}")
logger.info(f"LLM Model: {settings.llm_model}")

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Multi-tenant RAG system with advanced security and retrieval optimization",
    version="2.0.0",
    docs_url="/docs" if settings.environment == "development" else None,  # Disable docs in production
    redoc_url="/redoc" if settings.environment == "development" else None,
    lifespan=lifespan,
)

# Security Middleware (rate limiting, headers, validation)
app.add_middleware(SecurityMiddleware)

# Trusted Host Middleware (prevent host header attacks)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "api.yourdomain.com", "*.onrender.com", "*.hf.space", "*.ondigitalocean.app",
            "localhost", "127.0.0.1",
        ]
    )

# Configure CORS based on environment
if settings.environment == "production":
    # Production: Restrict to Vercel/Render/HF Spaces/DO deployments and specific origins
    allowed_origins = []
    # Allow all Vercel, Render, Hugging Face Space, and DigitalOcean deployments with regex
    allowed_origin_regex = r"https://(.*\.vercel\.app|.*\.onrender\.com|.*\.hf\.space|.*\.ondigitalocean\.app)"
    logger.info(f"CORS restricted to Vercel/Render/HF Spaces/DO deployments: {allowed_origin_regex}")
else:
    # Development: Allow localhost + Vercel deployments
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
    ]
    # Allow all Vercel preview deployments with regex
    allowed_origin_regex = r"https://rag-new-.*\.vercel\.app"
    logger.info("CORS enabled for local development + Vercel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    max_age=3600,
)

# Include routers
app.include_router(meta_router)
app.include_router(auth_router)
app.include_router(public_router)
app.include_router(clients_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(voice_router)
app.include_router(whatsapp_router)


@app.get("/widget.js")
async def widget_js():
    """Serve the embeddable chat widget script."""
    from pathlib import Path as _Path
    from fastapi.responses import FileResponse
    return FileResponse(
        _Path(__file__).parent / "static" / "widget.js",
        media_type="application/javascript",
    )


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "RAG Chatbot API is running"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG Chatbot API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
