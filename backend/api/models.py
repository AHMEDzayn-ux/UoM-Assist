"""
Pydantic models for API request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# Auth Models
class LoginRequest(BaseModel):
    """Operator console login."""
    email: str = Field(..., description="Operator email")
    password: str = Field(..., description="Operator password")


class RegisterRequest(BaseModel):
    """Create a new operator account."""
    email: str
    password: str = Field(..., min_length=6)
    name: str = ""


class LoginResponse(BaseModel):
    token: str
    email: str
    name: Optional[str] = None


class MeResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None
    is_superadmin: bool = False
    role: str = "operator"                 # superadmin | operator


# Domain template metadata (for the create-client domain picker)
class DomainInfo(BaseModel):
    key: str
    display_name: str
    persona: str
    bot_name: str
    greeting: str


class DomainListResponse(BaseModel):
    domains: List[DomainInfo]


# Client Management Models
class ClientCreate(BaseModel):
    """Request model for creating a new client (tenant)."""
    slug: str = Field(..., description="URL-safe unique identifier (used in /c/{slug})")
    name: str = Field(default="", description="Display name of the client")
    description: str = Field(default="", description="Optional description")
    domain: str = Field(default="generic", description="Vertical: telecom | university | generic")
    persona: Optional[str] = Field(default=None, description="Override persona (defaults to domain template)")
    bot_name: Optional[str] = Field(default=None, description="Assistant display name")
    greeting: Optional[str] = Field(default=None, description="Opening greeting shown to customers")
    accent_color: Optional[str] = Field(default=None, description="Brand accent color (hex)")

    class Config:
        json_schema_extra = {
            "example": {
                "slug": "abc-university",
                "name": "ABC University",
                "description": "Admissions & student services",
                "domain": "university"
            }
        }


class ClientUpdate(BaseModel):
    """Partial update of a client's configuration."""
    name: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    persona: Optional[str] = None
    bot_name: Optional[str] = None
    greeting: Optional[str] = None
    accent_color: Optional[str] = None
    wa_enabled: Optional[bool] = None
    wa_phone_number_id: Optional[str] = None
    wa_access_token: Optional[str] = None


class ClientResponse(BaseModel):
    """Response model for client information."""
    slug: str
    name: str = ""
    description: str = ""
    domain: str = "generic"
    persona: Optional[str] = None
    bot_name: str = "Assistant"
    greeting: Optional[str] = None
    accent_color: str = "#4f46e5"
    document_count: int = 0
    wa_enabled: bool = False
    wa_phone_number_id: Optional[str] = None


class ClientListResponse(BaseModel):
    """Response model for listing all clients."""
    clients: List[ClientResponse]
    total: int


# Public (customer-facing) models — scoped to a single client, no admin data
class PublicConfigResponse(BaseModel):
    """Branding/config the customer page + widget need. No secrets."""
    slug: str
    name: str
    bot_name: str
    greeting: str
    accent_color: str
    domain: str


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    detail: Optional[str] = None


# Document Management Models
class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    files_processed: int
    chunks_created: int
    total_documents: int
    chunk_previews: List[Dict[str, Any]] = Field(default_factory=list, description="Preview of created chunks")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Documents uploaded successfully",
                "files_processed": 2,
                "chunks_created": 45,
                "total_documents": 47,
                "chunk_previews": [
                    {
                        "chunk_index": 0,
                        "text_preview": "This is the beginning of the document...",
                        "chunk_size": 512,
                        "metadata": {"source": "document.pdf", "page": 1}
                    }
                ]
            }
        }


class DocumentInfo(BaseModel):
    """Information about a document chunk."""
    chunk_index: int
    text_preview: str
    metadata: Dict[str, Any]


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    client_id: str
    total_documents: int
    documents: List[DocumentInfo]


# Query & Chat Models
class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(default=6, ge=1, le=15, description="Number of documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Filter by metadata (e.g., {\"category\": \"data\", \"tags\": \"budget\"})")
    use_hybrid_search: bool = Field(default=True, description="Enable hybrid vector+keyword search")
    use_reranking: bool = Field(default=True, description="Enable cross-encoder re-ranking")
    use_query_normalization: bool = Field(default=True, description="Enable smart query normalization (abbreviations, typos, semantic expansion)")
    use_query_rewriting: bool = Field(default=False, description="Enable LLM-based query rewriting")
    use_hyde: bool = Field(default=False, description="Enable HyDE (Hypothetical Document Embeddings)")
    use_multi_query: bool = Field(default=False, description="Enable multi-query RAG fusion (generates query variations)")
    num_query_variations: int = Field(default=3, ge=2, le=5, description="Number of query variations for multi-query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the cheapest data plans?",
                "top_k": 3,
                "include_sources": True,
                "metadata_filter": {"category": "data", "tags": "budget"},
                "use_hybrid_search": True,
                "use_reranking": True,
                "use_query_normalization": True,
                "use_query_rewriting": False,
                "use_hyde": False,
                "use_multi_query": False,
                "num_query_variations": 3
            }
        }


class Source(BaseModel):
    """Source document information."""
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: Optional[List[Source]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The admission requirements include...",
                "sources": [
                    {
                        "text": "Excerpt from document...",
                        "metadata": {"source": "admission_guide.pdf"},
                        "distance": 0.85
                    }
                ]
            }
        }


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for conversational chat."""
    message: str = Field(..., description="The user's message")
    history: List[ChatMessage] = Field(default=[], description="Previous conversation history")
    session_id: Optional[str] = Field(default=None, description="Client-generated conversation id")
    use_retrieval: bool = Field(default=True, description="Whether to use document retrieval")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")
    use_hybrid_search: bool = Field(default=True, description="Enable hybrid vector+keyword search")
    use_reranking: bool = Field(default=True, description="Enable cross-encoder re-ranking")
    use_query_normalization: bool = Field(default=True, description="Enable smart query normalization (abbreviations, typos, semantic expansion)")
    use_query_rewriting: bool = Field(default=False, description="Enable LLM-based query rewriting")
    use_hyde: bool = Field(default=False, description="Enable HyDE (Hypothetical Document Embeddings)")
    use_multi_query: bool = Field(default=False, description="Enable multi-query RAG fusion (generates query variations)")
    num_query_variations: int = Field(default=3, ge=2, le=5, description="Number of query variations for multi-query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Tell me more about that",
                "history": [
                    {"role": "user", "content": "What are the requirements?"},
                    {"role": "assistant", "content": "The requirements include..."}
                ],
                "use_retrieval": True,
                "top_k": 3,
                "use_query_normalization": True,
                "use_multi_query": False,
                "num_query_variations": 3
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat."""
    response: str
    used_retrieval: bool
    sources: Optional[List[Source]] = None
    escalated: bool = False
    emotion: Optional[str] = None
    interaction_id: Optional[int] = None


# Learning-loop models
class FeedbackRequest(BaseModel):
    interaction_id: int
    rating: str = Field(..., description="'up' or 'down'")


class InsightsResponse(BaseModel):
    total_conversations: int
    total_turns: int
    deflection_rate: float
    escalation_rate: float
    satisfaction_rate: Optional[float] = None
    thumbs_up: int = 0
    thumbs_down: int = 0
    weak_count: int = 0
    emotion_breakdown: Dict[str, int] = Field(default_factory=dict)
    top_questions: List[Dict[str, Any]] = Field(default_factory=list)


class GapCluster(BaseModel):
    representative_question: str
    count: int
    examples: List[str]


class GapListResponse(BaseModel):
    gaps: List[GapCluster]


class DraftRequest(BaseModel):
    questions: List[str]


class DraftResponse(BaseModel):
    title: str
    content: str


class KbEntryRequest(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = None


# Escalation (human handoff) models
class EscalationResponse(BaseModel):
    id: int
    reason: str
    summary: Optional[str] = None
    emotion: Optional[str] = None
    intensity: Optional[int] = None
    transcript: Optional[str] = None
    status: str
    created_at: str


class EscalationListResponse(BaseModel):
    escalations: List[EscalationResponse]
    open_count: int


# Transactional actions
class ActionResponse(BaseModel):
    id: int
    action_type: str
    kind: str
    reference: Optional[str] = None
    payload: Optional[dict] = None
    result: Optional[str] = None
    status: str
    session_id: Optional[str] = None
    created_at: str


class ActionListResponse(BaseModel):
    actions: List[ActionResponse]
    open_count: int


class ActionStatusRequest(BaseModel):
    status: str  # open | done


class AccountResponse(BaseModel):
    id: int
    identifier: str
    name: Optional[str] = None
    data: Optional[dict] = None


class AccountListResponse(BaseModel):
    accounts: List[AccountResponse]
