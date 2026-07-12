"""
Client Management Router (admin-only)

DB-backed CRUD for tenants. Metadata (persona, domain, branding, WhatsApp) lives
in the database; vectors live in per-client FAISS collections. All endpoints
require admin authentication — customers use the public router instead.
"""

import re

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session

from api.models import (
    ClientCreate,
    ClientUpdate,
    ClientResponse,
    ClientListResponse,
    MessageResponse,
    EscalationResponse,
    EscalationListResponse,
    InsightsResponse,
    GapListResponse,
    GapCluster,
    DraftRequest,
    DraftResponse,
    KbEntryRequest,
    DocumentUploadResponse,
    ActionResponse,
    ActionListResponse,
    ActionStatusRequest,
    AccountResponse,
    AccountListResponse,
)
from services.rag_pipeline import MultiClientRAGPipeline
from services import client_store, learning, auth_service
from database import get_db
from db_models import User, Client
from auth import require_admin
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/clients", tags=["clients"])

_SLUG_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def owned_client(
    slug: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
) -> Client:
    """Auth + tenant isolation: resolve a client the current operator owns, else 404.

    Used as a dependency on every slug-scoped admin endpoint so one operator can
    never see or touch another operator's client.
    """
    client = client_store.get_client(db, slug)
    if client is None or (client.owner_id != user.id and not user.is_superadmin):
        raise HTTPException(status_code=404, detail=f"Client '{slug}' not found")
    return client

# Global pipeline manager (initialized on first use).
pipeline_manager: MultiClientRAGPipeline = None


def get_pipeline_manager() -> MultiClientRAGPipeline:
    global pipeline_manager
    if pipeline_manager is None:
        logger.info("Initializing MultiClientRAGPipeline manager")
        pipeline_manager = MultiClientRAGPipeline()
    return pipeline_manager


def _to_response(client, document_count: int) -> ClientResponse:
    return ClientResponse(
        slug=client.slug,
        name=client.name,
        description=client.description,
        domain=client.domain,
        persona=client.persona,
        bot_name=client.bot_name,
        greeting=client.greeting or "",
        accent_color=client.accent_color,
        document_count=document_count,
        wa_enabled=client.wa_enabled,
        wa_phone_number_id=client.wa_phone_number_id,
    )


def _doc_count(db: Session, slug: str) -> int:
    # Lightweight (DB-only) count; avoids loading heavy models just to list.
    return len(client_store.list_documents(db, slug))


@router.post("", response_model=ClientResponse, status_code=status.HTTP_201_CREATED)
async def create_client(
    payload: ClientCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    """Create a new client (tenant) with a domain-driven persona."""
    if not _SLUG_RE.match(payload.slug):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="slug must be URL-safe (letters, numbers, - and _ only)",
        )
    if client_store.get_client(db, payload.slug) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Client '{payload.slug}' already exists",
        )

    client = client_store.create_client(
        db,
        slug=payload.slug,
        name=payload.name,
        description=payload.description,
        domain=payload.domain,
        persona=payload.persona,
        bot_name=payload.bot_name,
        greeting=payload.greeting,
        accent_color=payload.accent_color,
        owner_id=user.id,
    )

    # Create the vector pipeline with resolved persona + domain.
    manager = get_pipeline_manager()
    manager.create_pipeline(
        client_id=client.slug,
        system_role=client_store.resolve_persona(client),
        domain=client.domain,
    )

    logger.info(f"Created client: {client.slug} (domain={client.domain})")
    return _to_response(client, 0)


@router.get("", response_model=ClientListResponse)
async def list_clients(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
):
    """List the current operator's clients (paginated)."""
    owner_filter = None if user.is_superadmin else user.id
    all_clients = client_store.list_clients(db, owner_id=owner_filter)
    total = len(all_clients)
    page = all_clients[skip:skip + limit]
    return ClientListResponse(
        clients=[_to_response(c, _doc_count(db, c.slug)) for c in page],
        total=total,
    )


@router.get("/{slug}", response_model=ClientResponse)
async def get_client(
    slug: str,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    client = client_store.get_client(db, slug)
    if client is None:
        raise HTTPException(status_code=404, detail=f"Client '{slug}' not found")
    return _to_response(client, _doc_count(db, slug))


@router.patch("/{slug}", response_model=ClientResponse)
async def update_client(
    slug: str,
    payload: ClientUpdate,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """Update a client's config. Reloads its pipeline persona/domain if changed."""
    client = client_store.update_client(db, slug, **payload.model_dump(exclude_unset=True))
    if client is None:
        raise HTTPException(status_code=404, detail=f"Client '{slug}' not found")

    # Refresh the in-memory pipeline so persona/domain changes take effect now.
    manager = get_pipeline_manager()
    if slug in manager.pipelines:
        pipe = manager.pipelines[slug]
        pipe.system_role = client_store.resolve_persona(client)
        from domain_templates import get_template
        pipe.domain = client.domain
        pipe.domain_template = get_template(client.domain)

    return _to_response(client, _doc_count(db, slug))


@router.delete("/{slug}", response_model=MessageResponse)
async def delete_client(
    slug: str,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    client = client_store.get_client(db, slug)
    if client is None:
        raise HTTPException(status_code=404, detail=f"Client '{slug}' not found")

    manager = get_pipeline_manager()
    if slug in manager.pipelines:
        manager.delete_pipeline(slug)  # removes files + in-memory pipeline
    else:
        client_store.delete_collection_files(slug)  # remove files without loading models

    client_store.delete_client(db, slug)  # removes DB row + documents (cascade)
    logger.info(f"Deleted client: {slug}")
    return MessageResponse(
        message=f"Client '{slug}' deleted successfully",
        detail="All associated data has been removed",
    )


def _escalation_to_response(e) -> EscalationResponse:
    return EscalationResponse(
        id=e.id,
        reason=e.reason,
        summary=e.summary,
        emotion=e.emotion,
        intensity=e.intensity,
        transcript=e.transcript,
        status=e.status,
        created_at=e.created_at.isoformat() if e.created_at else "",
    )


@router.get("/{slug}/escalations", response_model=EscalationListResponse)
async def list_escalations(
    slug: str,
    status_filter: str = None,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """List human-handoff escalations for a client (admin inbox)."""
    rows = client_store.list_escalations(db, slug, status=status_filter)
    open_count = sum(1 for r in rows if r.status == "open") if not status_filter else \
        len(client_store.list_escalations(db, slug, status="open"))
    return EscalationListResponse(
        escalations=[_escalation_to_response(e) for e in rows],
        open_count=open_count,
    )


@router.post("/{slug}/escalations/{escalation_id}/resolve", response_model=MessageResponse)
async def resolve_escalation(
    slug: str,
    escalation_id: int,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """Mark an escalation as resolved."""
    esc = client_store.resolve_escalation(db, escalation_id)
    if esc is None:
        raise HTTPException(status_code=404, detail="Escalation not found")
    return MessageResponse(message="Escalation resolved", detail=f"#{escalation_id}")


# ---- Learning loop (insights + knowledge gaps) ------------------------------

@router.get("/{slug}/insights", response_model=InsightsResponse)
async def get_insights(
    slug: str,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """Aggregated conversation analytics for a client."""
    return InsightsResponse(**learning.compute_insights(db, slug))


@router.get("/{slug}/gaps", response_model=GapListResponse)
async def get_gaps(
    slug: str,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """Clustered unanswered questions (knowledge gaps)."""
    pipeline = get_pipeline_manager().get_pipeline(slug)
    if pipeline is None:
        return GapListResponse(gaps=[])
    clusters = learning.cluster_gaps(db, slug, pipeline.embeddings_service)
    # Self-heal: drop clusters the CURRENT knowledge base can now answer
    # (e.g. after the operator taught it) so fixed gaps disappear.
    open_clusters = []
    for c in clusters:
        try:
            if pipeline._retrieve_context(c["representative_question"], top_k=1):
                continue  # KB can now answer it → no longer a gap
        except Exception:
            pass
        open_clusters.append(c)
    return GapListResponse(gaps=[GapCluster(**c) for c in open_clusters])


@router.post("/{slug}/gaps/draft", response_model=DraftResponse)
async def draft_gap_answer(
    slug: str,
    payload: DraftRequest,
    _owned: Client = Depends(owned_client),
):
    """Draft a KB entry answering a cluster of unanswered questions (on-demand LLM call)."""
    pipeline = get_pipeline_manager().get_pipeline(slug)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Client '{slug}' not found")
    draft = pipeline.draft_kb_entry(payload.questions)
    return DraftResponse(title=draft["title"], content=draft["content"])


@router.post("/{slug}/kb-entry", response_model=DocumentUploadResponse)
async def add_kb_entry(
    slug: str,
    payload: KbEntryRequest,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """Add an approved KB entry to the client's knowledge base (closes the loop)."""
    pipeline = get_pipeline_manager().get_pipeline(slug)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Client '{slug}' not found")
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    chunks = pipeline.add_kb_entry(payload.title, payload.content, payload.tags)
    try:
        client_store.add_document(db, client_slug=slug, filename=f"[learned] {payload.title}",
                                  doc_type="learned", chunk_count=chunks)
    except Exception as e:
        logger.warning(f"Could not record learned doc row: {e}")
    return DocumentUploadResponse(
        message="Knowledge base updated",
        files_processed=1,
        chunks_created=chunks,
        total_documents=0,
        chunk_previews=[],
    )


# ---- Transactional actions (requests inbox + mock accounts) ------------------

def _action_to_response(a) -> ActionResponse:
    return ActionResponse(
        id=a.id,
        action_type=a.action_type,
        kind=a.kind,
        reference=a.reference,
        payload=a.payload or {},
        result=a.result,
        status=a.status,
        session_id=a.session_id,
        created_at=a.created_at.isoformat() if a.created_at else "",
    )


@router.get("/{slug}/requests", response_model=ActionListResponse)
async def list_requests(
    slug: str,
    status_filter: str = None,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """List transactional actions the agent has taken (tickets, callbacks, changes)."""
    rows = client_store.list_action_requests(db, slug, status=status_filter)
    open_count = sum(1 for r in rows if r.status == "open") if not status_filter else \
        len(client_store.list_action_requests(db, slug, status="open"))
    return ActionListResponse(
        actions=[_action_to_response(a) for a in rows],
        open_count=open_count,
    )


@router.post("/{slug}/requests/{action_id}/status", response_model=MessageResponse)
async def set_request_status(
    slug: str,
    action_id: int,
    payload: ActionStatusRequest,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """Mark an action request open/done."""
    row = client_store.set_action_status(db, action_id, payload.status)
    if row is None:
        raise HTTPException(status_code=404, detail="Action not found")
    return MessageResponse(message=f"Marked {payload.status}", detail=f"#{action_id}")


@router.get("/{slug}/accounts", response_model=AccountListResponse)
async def list_accounts(
    slug: str,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """List the seeded mock accounts (so the operator knows the demo identifiers)."""
    rows = client_store.list_mock_accounts(db, slug)
    return AccountListResponse(accounts=[
        AccountResponse(id=r.id, identifier=r.identifier, name=r.name, data=r.data or {})
        for r in rows
    ])


@router.post("/{slug}/accounts/seed", response_model=AccountListResponse)
async def seed_accounts(
    slug: str,
    db: Session = Depends(get_db),
    _owned: Client = Depends(owned_client),
):
    """Seed domain-appropriate demo accounts for account lookup/change demos."""
    client = client_store.get_client(db, slug)
    if client is None:
        raise HTTPException(status_code=404, detail=f"Client '{slug}' not found")
    rows = client_store.seed_demo_accounts(db, slug, client.domain)
    return AccountListResponse(accounts=[
        AccountResponse(id=r.id, identifier=r.identifier, name=r.name, data=r.data or {})
        for r in rows
    ])
