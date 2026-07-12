"""
RAG Pipeline Service
Orchestrates the complete RAG workflow: document processing, retrieval, and generation
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from services.document_loader import DocumentLoader
from services.embeddings import EmbeddingsService
from services.vector_store import VectorStoreService
from services.llm_service import LLMService
from services.retrieval_optimizer import RetrievalOptimizer
from logger import get_logger
from config import get_settings
from domain_templates import get_template

logger = get_logger(__name__)
settings = get_settings()


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline.
    
    Orchestrates:
    1. Document loading and chunking
    2. Embedding generation
    3. Vector storage and retrieval
    4. LLM-based response generation
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        api_key: Optional[str] = None,
        system_role: Optional[str] = None,
        domain: Optional[str] = None,
        enable_advanced_retrieval: bool = True,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize the RAG pipeline.

        Args:
            collection_name: Name of the vector store collection
            api_key: Groq API key (optional, uses settings if not provided)
            system_role: System role for LLM responses (e.g., "university advisor")
            domain: Vertical key (telecom/university/generic) driving domain
                    template defaults (normalization context, few-shot examples)
            enable_advanced_retrieval: Enable advanced retrieval optimization
            vector_weight: Weight for vector search in hybrid (0-1)
            keyword_weight: Weight for keyword search in hybrid (0-1)
        """
        self.collection_name = collection_name
        # Domain template drives normalization context + few-shot examples so
        # each vertical answers in-character with no cross-domain bleed.
        self.domain = domain or "generic"
        self.domain_template = get_template(self.domain)
        self.system_role = system_role or self.domain_template.persona
        self.enable_advanced_retrieval = enable_advanced_retrieval
        
        # Initialize all services
        self.doc_loader = DocumentLoader()
        self.embeddings_service = EmbeddingsService()
        self.vector_store = VectorStoreService()
        self.llm_service = LLMService(api_key=api_key)
        
        # Initialize advanced retrieval optimizer
        if enable_advanced_retrieval:
            self.retrieval_optimizer = RetrievalOptimizer(
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                enable_reranking=True
            )
            logger.info("Advanced retrieval optimization ENABLED")
        else:
            self.retrieval_optimizer = None
            logger.info("Advanced retrieval optimization DISABLED")
        
        logger.info(f"RAGPipeline initialized for collection: {collection_name}")

    # Common greetings / pleasantries that should NOT trigger document retrieval.
    _SMALLTALK = {
        "hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "yo", "hiya",
        "good morning", "good afternoon", "good evening", "gm", "ge",
        "thanks", "thank you", "thankyou", "thx", "ty", "cheers",
        "ok", "okay", "cool", "nice", "great", "bye", "goodbye", "see you",
        "how are you", "how are you?", "whatsup", "what's up", "sup",
    }

    def _is_smalltalk(self, message: str) -> bool:
        """True for short greetings/pleasantries with no informational intent."""
        if not message:
            return True
        text = message.strip().lower().rstrip("!.?")
        if text in self._SMALLTALK:
            return True
        # Very short (1-2 words) and starts with a greeting token.
        words = text.split()
        if len(words) <= 2 and words and words[0] in {
            "hi", "hello", "hey", "yo", "thanks", "thank", "bye", "hiya", "sup"
        }:
            return True
        return False

    # Signals worth running the (paid) emotion classifier for. Plain neutral
    # questions have none of these, so we skip the extra call.
    _EMOTION_SIGNAL_WORDS = (
        "useless", "terrible", "worst", "ridiculous", "annoy", "frustrat", "angry",
        "unacceptable", "complaint", "complain", "refund", "cancel", "manager",
        "human", "agent", "supervisor", "speak to", "talk to", "real person",
        "stupid", "hate", "sick of", "fed up", "waste", "scam", "cheat", "rip off",
        "not working", "doesn't work", "still not", "again", "third time", "3rd time",
        "urgent", "asap", "immediately", "disappointed", "poor service", "never",
        "fk", "wtf", "damn", "shit", "hell",
    )

    def _needs_emotion_check(self, message: str) -> bool:
        """Cheap heuristic: only classify emotion when the message shows signals."""
        if not message:
            return False
        text = message.lower()
        if any(w in text for w in self._EMOTION_SIGNAL_WORDS):
            return True
        # Emphatic punctuation
        if "!!" in message or "??" in message or "?!" in message or message.count("!") >= 2:
            return True
        # An ALL-CAPS word (shouting), 4+ letters
        for word in message.split():
            stripped = "".join(ch for ch in word if ch.isalpha())
            if len(stripped) >= 4 and stripped.isupper():
                return True
        return False

    def rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 keyword index from the loaded collection's documents.

        BM25 lives in memory and is only built at index time; when a pipeline is
        lazy-loaded from disk it must be rebuilt or hybrid search silently
        degrades to vector-only.
        """
        if not self.retrieval_optimizer:
            return
        try:
            collection = self.vector_store.collections.get(self.collection_name)
            docs = (collection or {}).get('documents', [])
            if docs:
                self.retrieval_optimizer.build_bm25_index(
                    collection_name=self.collection_name,
                    documents=docs,
                    doc_ids=list(range(len(docs))),
                )
                logger.info(f"Rebuilt BM25 index for {self.collection_name} ({len(docs)} docs)")
        except Exception as e:
            logger.warning(f"Could not rebuild BM25 index for {self.collection_name}: {e}")

    # ===================== AGENTIC ANSWERING =====================

    def _retrieve_context(
        self,
        query: str,
        top_k: int = 4,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Relevance-gated retrieval used as the agent's search tool.

        Returns a list of relevant docs, or an EMPTY list when nothing is
        genuinely relevant (so the agent can honestly say 'I don't have that'
        instead of being force-fed an off-topic document).
        """
        try:
            query_embedding = self.embeddings_service.embed_text(query)
            initial_k = top_k * 10 if (self.retrieval_optimizer and (use_hybrid_search or use_reranking)) else top_k
            results = self.vector_store.query(
                collection_name=self.collection_name,
                query_embeddings=[query_embedding],
                n_results=initial_k,
            )
            if not (results and results.get('documents') and results['documents'][0]):
                return []

            texts = results['documents'][0]
            metas = (results.get('metadatas') or [[]])[0]
            dists = (results.get('distances') or [[]])[0]

            retrieved = []
            for i, text in enumerate(texts):
                meta = metas[i] if i < len(metas) else {}
                if meta.get('content_type') == 'question':
                    continue
                retrieved.append({
                    'text': text,
                    'metadata': meta,
                    'distance': float(dists[i]) if i < len(dists) else 0.0,
                    'doc_id': i,
                })
            if not retrieved:
                return []

            # Relevance gate: if even the closest match is too far, nothing is relevant.
            best = min(d['distance'] for d in retrieved)
            if best > settings.hard_distance_cutoff:
                logger.info(f"Search '{query[:40]}': best distance {best:.3f} > cutoff; no relevant context")
                return []

            filtered = [d for d in retrieved if d['distance'] <= settings.distance_threshold]
            if not filtered:
                filtered = sorted(retrieved, key=lambda x: x['distance'])[:settings.min_results_after_filter]

            # Hybrid + rerank for precision
            if self.retrieval_optimizer and (use_hybrid_search or use_reranking):
                try:
                    optimized, _ = self.retrieval_optimizer.optimize_retrieval(
                        collection_name=self.collection_name,
                        query=query,
                        vector_results=filtered,
                        llm_service=self.llm_service,
                        use_hybrid=use_hybrid_search,
                        use_reranking=use_reranking,
                        use_query_rewriting=False,
                        use_hyde=False,
                        top_k_initial=50,
                        top_k_final=top_k,
                    )
                    filtered = optimized
                except Exception as e:
                    logger.warning(f"Optimization failed, using vector results: {e}")

            return filtered[:top_k]
        except Exception as e:
            logger.warning(f"Retrieval error for '{query[:40]}': {e}")
            return []

    def _agent_system_prompt(self) -> str:
        """Lean agent prompt: persona + genuine capability boundaries (no case-patches)."""
        return f"""You are {self.system_role}

Today's date is {datetime.now():%Y-%m-%d}. Resolve any relative date the customer gives ("yesterday", "10th of July", "last Monday") against this before calling a tool that takes a date.

🌐 LANGUAGE (very important):
- Reply in the customer's language. If their latest message is in Sinhala script (සිංහල) or in romanized Sinhala ("Singlish", e.g. "mata plan eka gana danaganna oney"), reply in natural, fluent Sinhala script. If it is in Tamil script (தமிழ்) or romanized Tamil, reply in natural, fluent Tamil script. If it is in English, reply in English. Never switch the language on the customer.
- The customer's latest message is the ONLY thing that sets your reply language. Retrieved documents may be in Sinhala, Tamil, or English — this NEVER changes your reply language. If the message is English, reply in English even when every source chunk is Sinhala; translate the meaning across rather than echoing the document's script.
- BUT always write your search_knowledge_base queries in ENGLISH — the knowledge base is mostly English. Translate the customer's need into a focused English query, even when the conversation is in Sinhala or Tamil. Then answer them back in their own language.

You are a customer-care agent. Operate by these principles:

- Use the search_knowledge_base tool for ANY question about products, plans, prices, policies, features, coverage, or how to do something ("how do I…", "can I…", "what is…"). When unsure whether we cover it, SEARCH FIRST rather than giving generic advice. Base every factual claim ONLY on what the tool returns.
- If a search returns no relevant information, tell the customer you don't have that detail and offer to connect them to a human. NEVER invent prices, policies, or facts.
- You have action tools to actually HELP — you are not limited to a fixed script. Use whichever tool fits what the customer is asking — don't guess the answer yourself. Collect the details a tool needs (ask for anything missing) before calling it. When a tool returns a reference number, give it to the customer.
- You can look up or change an account ONLY through the account tools, and ONLY when the customer provides an identifier (phone number / email / account or application ID). NEVER invent, assume, or guess account details — if you don't have the tool result, you don't know it. Before making any CHANGE to an account, state the exact change and get the customer's clear confirmation ("yes") first.
- For greetings, thanks, or small talk, reply warmly and briefly — do NOT search or call tools.
- If the customer reports a vague problem ("it's not working", "I have an issue"), ask what specifically is wrong BEFORE troubleshooting — don't guess.
- Answer ONLY what was asked. Be concise: lead with the key points, then offer more. Don't dump unrelated details.
- Be warm, confident, and human. Use short paragraphs and bullet points (•) when listing.
- Do NOT use emojis in your replies — they get read aloud in voice mode. Use plain text only."""

    # Tool schema exposed to the model (OpenAI function-calling format).
    _AGENT_TOOLS = [{
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company knowledge base for facts needed to answer the "
                "customer (plans, prices, policies, features, coverage, procedures). "
                "Do NOT call this for greetings, thanks, or small talk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query for what to look up",
                    }
                },
                "required": ["query"],
            },
        },
    }]

    _ESCALATE_TOOL = {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Hand this conversation off to a human agent. Use when the customer is "
                "very upset or angry, explicitly asks for a human/manager, or you cannot "
                "resolve their issue. Provide a short reason and a one-paragraph summary "
                "of the customer's issue for the human agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Short reason for escalation"},
                    "summary": {"type": "string", "description": "One-paragraph summary of the issue for the human agent"},
                },
                "required": ["reason"],
            },
        },
    }

    @property
    def client_slug(self) -> str:
        if self.collection_name.startswith("client_"):
            return self.collection_name[len("client_"):]
        return self.collection_name

    def _mood_directive(self, emotion: Dict[str, Any]) -> str:
        """Turn a detected mood into tone guidance appended to the agent prompt."""
        e = (emotion or {}).get("emotion", "neutral")
        i = (emotion or {}).get("intensity", 1)
        wants_human = (emotion or {}).get("wants_human", False)
        lines = []
        if e == "angry":
            lines.append("The customer sounds ANGRY. Open with a sincere, brief apology. Stay calm and non-defensive. Focus entirely on resolving their issue. Do not be cheerful or salesy.")
        elif e == "frustrated":
            lines.append("The customer sounds FRUSTRATED. Acknowledge their frustration, be extra clear and efficient, and don't make them repeat themselves.")
        elif e == "confused":
            lines.append("The customer sounds CONFUSED. Slow down, explain simply and step by step, and check they're following.")
        elif e == "happy":
            lines.append("The customer is in a good mood — match their warmth.")
        if wants_human:
            lines.append("The customer explicitly asked for a human — call the escalate_to_human tool NOW (write a short summary of the conversation), then warmly tell them a human agent will follow up. Do not just ask for more details first.")
        elif e in ("angry", "frustrated") and i >= 4:
            lines.append("If you cannot fully resolve this, sincerely offer to connect them to a human agent and use the escalate_to_human tool.")
        if not lines:
            return ""
        return "\n\nCUSTOMER MOOD (adapt your tone):\n- " + "\n- ".join(lines)

    def _record_escalation(self, reason, summary, emotion, conversation_history, message) -> None:
        """Persist a human-handoff record with a transcript snippet."""
        try:
            from database import SessionLocal
            from services.client_store import create_escalation
            transcript = [f"{m.get('role')}: {m.get('content')}" for m in (conversation_history or [])[-8:]]
            transcript.append(f"user: {message}")
            db = SessionLocal()
            try:
                create_escalation(
                    db,
                    client_slug=self.client_slug,
                    reason=reason,
                    summary=summary or "",
                    emotion=(emotion or {}).get("emotion"),
                    intensity=(emotion or {}).get("intensity"),
                    transcript="\n".join(transcript),
                )
            finally:
                db.close()
            logger.info(f"Escalation recorded for {self.client_slug}: {reason}")
        except Exception as e:
            logger.warning(f"Failed to record escalation: {e}")

    def agent_chat(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4,
        max_iterations: int = 3,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Agentic answering: the model reasons and decides when to search the
        knowledge base. Falls back to a direct grounded answer if the model
        produces a malformed tool call (smaller models occasionally do), so the
        customer never sees an error.
        """
        conversation_history = conversation_history or []
        if self.llm_service.llm is None:
            return {'answer': "The assistant is temporarily unavailable.", 'sources': [],
                    'used_retrieval': False, 'emotion': {'emotion': 'neutral', 'intensity': 1}, 'escalated': False}

        # Detect the customer's mood — but only when it's worth a call. Small talk
        # and plainly neutral questions skip the classifier entirely (saves ~1
        # LLM call/turn); we only classify when the message shows emotional signals.
        if self._is_smalltalk(message) or not self._needs_emotion_check(message):
            emotion = {"emotion": "neutral", "intensity": 1, "wants_human": False}
        else:
            emotion = self.llm_service.detect_emotion(message)

        try:
            result = self._run_agent_loop(message, conversation_history, top_k, max_iterations, emotion, session_id)
        except Exception as e:
            logger.warning(f"Agent loop failed ({e}); using direct fallback")
            result = self._fallback_answer(message, conversation_history, top_k)

        result['emotion'] = emotion
        result.setdefault('escalated', False)
        result.setdefault('no_kb_match', False)

        # Safety net: an explicit request for a human always creates a handoff,
        # even if the model didn't call the escalate tool itself.
        if emotion.get('wants_human') and not result.get('escalated'):
            self._record_escalation(
                reason="Customer explicitly requested a human agent",
                summary=f"Customer asked to speak with a human. Latest message: {message}",
                emotion=emotion,
                conversation_history=conversation_history,
                message=message,
            )
            result['escalated'] = True

        return result

    def _fallback_answer(self, message, conversation_history, top_k=4) -> Dict[str, Any]:
        """Direct retrieve-then-answer path when the agent model is unavailable.

        Relevance-gated and restrained so it doesn't over-share like a dumb RAG:
        greetings/thanks get no retrieval, and it never volunteers unasked info.
        """
        if self._is_smalltalk(message):
            docs = []
        else:
            # Translate the query to the KB language (English) so Sinhala/Singlish
            # messages still retrieve — retrieval uses the converted query, the
            # answer below still uses the customer's original message + language.
            search_query = message
            if self.retrieval_optimizer:
                search_query = self.retrieval_optimizer.normalize_and_enhance_query(
                    query=message,
                    llm_service=self.llm_service,
                    domain_context=self.domain_template.normalization_context,
                )
            docs = self._retrieve_context(search_query, top_k=top_k)
        context = [d['text'] for d in docs] if docs else None
        system_prompt = (
            f"You are {self.system_role}. Answer ONLY the customer's exact question. "
            "Reply in the customer's language: if their message is in Sinhala or romanized "
            "Sinhala (Singlish), reply in natural Sinhala script; if in Tamil or romanized "
            "Tamil, reply in natural Tamil script; if in English, reply in English. "
            "Base facts strictly on the provided context; if there is no context, do NOT invent "
            "anything. NEVER volunteer plans, prices, products, or details the customer did not "
            "ask about. For greetings or thanks, just reply warmly in one line and ask how you can "
            "help — do not mention any plan. If the answer isn't in the context, say you don't have "
            "that detail and offer a human agent. You cannot access the customer's personal account, "
            "plan, or number — never claim to. Be concise and warm."
        )
        # Use the cheap utility model for the fallback so customers still get an
        # answer even when the main model is rate-limited or misfires.
        answer = self.llm_service.generate_response(
            query=message,
            context=context,
            system_prompt=system_prompt,
            conversation_history=conversation_history[-6:] if conversation_history else None,
            lightweight=True,
        )
        return {
            'answer': self.llm_service._clean_citation_phrases(answer),
            'sources': self._format_sources_for_citations(docs) if docs else [],
            'used_retrieval': bool(docs),
            'no_kb_match': (not self._is_smalltalk(message)) and not docs,
        }

    @staticmethod
    def _msg_text(content: Any) -> str:
        """Normalize LLM message content to a string (Gemini returns a list of parts)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    parts.append(p.get("text") or p.get("content") or "")
            return "".join(parts)
        return str(content) if content is not None else ""

    def _run_agent_loop(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        top_k: int,
        max_iterations: int,
        emotion: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from services import actions
        # Agent gets search + escalate + this domain's transactional actions.
        domain = getattr(self, "domain", "generic")
        llm_with_tools = self.llm_service.llm.bind_tools(
            [self._AGENT_TOOLS[0], self._ESCALATE_TOOL] + actions.get_action_tools(domain)
        )
        system_prompt = self._agent_system_prompt() + self._mood_directive(emotion)

        messages: List[Any] = [SystemMessage(content=system_prompt)]
        for m in conversation_history[-6:]:
            role, content = m.get('role'), m.get('content', '')
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=message))

        collected: List[Dict[str, Any]] = []
        escalated = False
        search_attempted = False

        for _ in range(max_iterations):
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            tool_calls = getattr(ai_msg, 'tool_calls', None) or []

            if not tool_calls:
                answer = self._msg_text(ai_msg.content).strip() or "Could you rephrase that?"
                return {
                    'answer': self.llm_service._clean_citation_phrases(answer),
                    'sources': self._format_sources_for_citations(collected) if collected else [],
                    'used_retrieval': bool(collected),
                    'escalated': escalated,
                    'no_kb_match': search_attempted and not collected,
                }

            for tc in tool_calls:
                name = tc.get('name')
                args = tc.get('args', {}) or {}
                tc_id = tc.get('id')
                if name == 'search_knowledge_base':
                    search_attempted = True
                    q = (args.get('query') or message).strip()
                    docs = self._retrieve_context(q, top_k=top_k)
                    collected.extend(docs)
                    if docs:
                        tool_text = "\n\n---\n\n".join(d['text'] for d in docs)
                    else:
                        tool_text = "No relevant information was found in the knowledge base for this query."
                    messages.append(ToolMessage(content=tool_text, tool_call_id=tc_id))
                elif name == 'escalate_to_human':
                    reason = (args.get('reason') or "Customer needs human assistance").strip()
                    summary = (args.get('summary') or "").strip()
                    self._record_escalation(reason, summary, emotion, conversation_history, message)
                    escalated = True
                    messages.append(ToolMessage(
                        content=("Escalation created — a human agent has been notified and will follow up. "
                                 "Warmly reassure the customer that a human will reach out."),
                        tool_call_id=tc_id,
                    ))
                elif actions.is_action(domain, name):
                    result_text = actions.execute_action(
                        self.client_slug, session_id, name, args, domain
                    )
                    messages.append(ToolMessage(content=result_text, tool_call_id=tc_id))
                else:
                    messages.append(ToolMessage(content="Unknown tool.", tool_call_id=tc_id))

        # Hit the iteration cap — force a final answer using what we have.
        final = self.llm_service.llm.invoke(messages)
        return {
            'answer': self.llm_service._clean_citation_phrases(self._msg_text(final.content).strip() or "Let me connect you with a human agent."),
            'sources': self._format_sources_for_citations(collected) if collected else [],
            'used_retrieval': bool(collected),
            'escalated': escalated,
            'no_kb_match': search_attempted and not collected,
        }
    
    def draft_kb_entry(self, questions: List[str]) -> Dict[str, str]:
        """Draft a KB card answering a cluster of unanswered questions (on-demand LLM call).

        The draft is a STARTING POINT — it uses [SPECIFY] placeholders for facts it
        doesn't know, which the operator fills/verifies before approving.
        """
        import json
        import re
        qlist = "\n".join(f"- {q}" for q in questions[:8])
        system = (
            "You help build a customer-support knowledge base. Given questions customers asked "
            "that we could NOT answer, draft ONE concise knowledge-base entry that would answer them. "
            "Respond with ONLY JSON: {\"title\": \"...\", \"content\": \"...\"}. "
            "Write clear, helpful content. For any specific fact you cannot know (prices, numbers, "
            "policies, dates), insert a [SPECIFY] placeholder — do NOT invent specifics. "
            "The human operator will fill placeholders and verify before publishing."
        )
        try:
            answer = self.llm_service.generate_response(
                query=f"Questions we couldn't answer:\n{qlist}",
                system_prompt=system,
            )
            m = re.search(r"\{.*\}", answer, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            return {
                "title": str(data.get("title", "")).strip() or (questions[0] if questions else "New topic"),
                "content": str(data.get("content", "")).strip(),
            }
        except Exception as e:
            logger.warning(f"draft_kb_entry failed: {e}")
            return {"title": questions[0] if questions else "New topic", "content": ""}

    def add_kb_entry(self, title: str, content: str, tags: Optional[List[str]] = None) -> int:
        """Ingest a single approved KB card into this client's collection. Returns chunk count."""
        import json
        import os
        import tempfile
        card = {
            "title": title,
            "category": "learned",
            "tags": tags or [],
            "content": content,
        }
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "kb_entry.json")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump([card], f)
            result = self.index_documents(
                file_paths=[tmp_path],
                metadata={"category": "learned", "doc_type": "learned"},
            )
            return result.get("chunks_created", result.get("total_chunks", 0)) if isinstance(result, dict) else 0
        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

    def index_documents(
        self,
        pdf_paths: List[str] = None,
        file_paths: List[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_parent_child: bool = False,
        generate_qa_pairs: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents into the vector store with advanced strategies.
        
        Supports:
        - PDF files (.pdf) - Text extraction and chunking
        - JSON files (.json) - Customer care FAQs, packages, catalogs
        
        Complete workflow:
        1. Load documents and chunk text (auto-detects PDF vs JSON)
        2. Generate QA pairs if enabled (for better search alignment)
        3. Generate embeddings for chunks/questions
        4. Store in vector database with rich metadata
        
        Args:
            pdf_paths: Legacy parameter - List of PDF file paths (deprecated, use file_paths)
            file_paths: List of file paths to index (PDF or JSON)
            chunk_size: Custom chunk size (optional)
            chunk_overlap: Custom chunk overlap (optional)
            metadata: Additional metadata to attach to all documents
            use_parent_child: Use parent-child chunking strategy (PDF only)
            generate_qa_pairs: Generate hypothetical QA pairs for better search
        
        Returns:
            Dictionary with indexing statistics
        """
        # Backward compatibility: support both pdf_paths and file_paths
        paths_to_process = file_paths if file_paths is not None else (pdf_paths or [])
        
        logger.info(f"Starting document indexing for {len(paths_to_process)} files")
        logger.info(f"Parent-child: {use_parent_child}, QA generation: {generate_qa_pairs}")
        
        all_chunks = []
        all_texts = []
        all_metadatas = []
        parent_lookup = {}  # Maps child_id to parent_text
        
        # Step 1: Load and chunk all documents
        for file_path in paths_to_process:
            try:
                logger.info(f"Processing: {file_path}")
                
                # Detect file type
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext == '.json':
                    # JSON files: Direct chunking (no parent-child or QA generation yet)
                    chunks = self.doc_loader.load_and_chunk_json(file_path, metadata=metadata)
                    logger.info(f"Loaded JSON: {len(chunks)} chunks created")
                    
                elif file_ext == '.pdf':
                    # Load PDF content
                    text = self.doc_loader.load_pdf(file_path)
                    
                    if use_parent_child:
                        # Use parent-child strategy
                        result = self.doc_loader.chunk_with_parent_child(text, metadata)
                        chunks = result['child_chunks']  # Index children for search
                        
                        # Store parent lookup for retrieval
                        for parent in result['parent_chunks']:
                            parent_id = parent['metadata']['parent_id']
                            parent_lookup[parent_id] = parent['text']
                        
                        logger.info(f"Created {len(result['parent_chunks'])} parents, {len(chunks)} children")
                        
                    elif generate_qa_pairs:
                        # Generate QA pairs for better search
                        chunks = self.doc_loader.chunk_with_qa_generation(
                            text, 
                            metadata,
                            llm_service=self.llm_service,
                            generate_qa=True
                        )
                        logger.info(f"Generated QA pairs for {len(chunks)} chunks")
                    else:
                        # Standard chunking
                        if chunk_size and chunk_overlap:
                            self.doc_loader.chunk_size = chunk_size
                            self.doc_loader.chunk_overlap = chunk_overlap
                        
                        chunks = self.doc_loader.chunk_text(text, metadata)
                
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}. Supported: .pdf, .json")
                
                # Prepare data
                for i, chunk in enumerate(chunks):
                    text = chunk['text']
                    chunk_metadata = chunk.get('metadata', {})
                    
                    # Add custom metadata
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    # Add document source (usually already set by loader, but ensure it's there)
                    if 'source' not in chunk_metadata:
                        chunk_metadata['source'] = file_path
                    chunk_metadata['chunk_index'] = i
                    
                    # Store parent lookup if using parent-child
                    if use_parent_child and 'parent_id' in chunk_metadata:
                        chunk_metadata['has_parent'] = True
                    
                    all_chunks.append(chunk)
                    all_texts.append(text)
                    all_metadatas.append(chunk_metadata)
                    
                    # If QA pairs generated, also index the questions
                    if generate_qa_pairs and 'generated_questions' in chunk:
                        for q_idx, question in enumerate(chunk['generated_questions']):
                            qa_metadata = chunk_metadata.copy()
                            qa_metadata['content_type'] = 'question'
                            qa_metadata['original_chunk_index'] = i
                            qa_metadata['question_index'] = q_idx
                            
                            all_texts.append(question)
                            all_metadatas.append(qa_metadata)
                            logger.debug(f"Added question: {question[:50]}...")
                
                logger.info(f"Processed {len(chunks)} chunks from {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                raise
        
        # Step 2: Generate embeddings
        logger.info(f"Generating embeddings for {len(all_texts)} chunks")
        embeddings = self.embeddings_service.embed_batch(all_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 3: Create collection if it doesn't exist
        if self.collection_name not in self.vector_store.list_collections():
            self.vector_store.create_collection(
                self.collection_name,
                embedding_dimension=settings.embedding_dimension
            )
            logger.info(f"Created collection: {self.collection_name}")
        
        # Step 4: Add documents to vector store
        self.vector_store.add_documents(
            collection_name=self.collection_name,
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas
        )
        
        # Step 4.5: Build BM25 index for hybrid search
        if self.retrieval_optimizer:
            logger.info("Building BM25 index for hybrid search...")
            self.retrieval_optimizer.build_bm25_index(
                collection_name=self.collection_name,
                documents=all_texts,
                doc_ids=list(range(len(all_texts)))
            )
            logger.info("BM25 index built successfully")
        
        # Step 5: Persist to disk
        self.vector_store.persist()
        logger.info("Vector store persisted to disk")
        
        # Create chunk previews for response
        chunk_previews = []
        for i, chunk in enumerate(all_chunks[:50]):  # Limit to first 50 chunks for response size
            text = chunk['text']
            preview = {
                'chunk_index': i,
                'text_preview': text,  # Full text, not truncated
                'chunk_size': len(text),
                'metadata': chunk.get('metadata', {})
            }
            chunk_previews.append(preview)
        
        logger.info(f"Created {len(chunk_previews)} chunk previews")
        
        # Return statistics
        stats = {
            'pdfs_processed': len(paths_to_process),  # Fixed: use paths_to_process instead of pdf_paths
            'total_chunks': len(all_chunks),
            'total_embeddings': len(embeddings),
            'collection_name': self.collection_name,
            'vector_store_count': self.vector_store.get_collection_count(self.collection_name),
            'chunk_previews': chunk_previews
        }
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        use_query_normalization: bool = True,
        use_query_rewriting: bool = False,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        num_query_variations: int = 3
    ) -> Dict[str, Any]:
        """
        Query the RAG system with advanced retrieval strategies.
        
        Complete workflow:
        1. Optional: Query normalization (default - fixes typos, expands abbreviations)
        2. Optional: Multi-query retrieval (OR query transformation with rewriting/HyDE)
        3. Generate embedding for question
        4. Retrieve relevant documents from vector store (with metadata filtering)
        5. Optional: Hybrid search (vector + BM25)
        6. Optional: Re-ranking with cross-encoder
        7. Filter by section if query intent detected
        8. Retrieve parent chunks if using parent-child strategy
        9. Generate response using LLM with context
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve (default from settings)
            return_sources: Whether to include source documents in response
            metadata_filter: Optional metadata filter (e.g., {"user_tier": "enterprise"})
            use_hybrid_search: Enable hybrid vector+keyword search
            use_reranking: Enable cross-encoder re-ranking
            use_query_normalization: Enable smart query normalization (default: True)
            use_query_rewriting: Enable query rewriting
            use_hyde: Enable HyDE (hypothetical document embeddings)
            use_multi_query: Enable multi-query RAG fusion (more expensive)
            num_query_variations: Number of query variations for multi-query (default: 3)
        
        Returns:
            Dictionary with answer and optional sources
        """
        logger.info(f"Processing query: {question}")
        logger.info(f"Advanced features: normalize={use_query_normalization}, hybrid={use_hybrid_search}, "
                   f"rerank={use_reranking}, rewrite={use_query_rewriting}, hyde={use_hyde}, multi_query={use_multi_query}")
        
        # Build metadata filter
        combined_filter = metadata_filter.copy() if metadata_filter else {}
        top_k = top_k or settings.retrieval_top_k
        optimization_metadata = {'transformations': []}
        
        # Step 0: Query normalization (lightweight preprocessing - default enabled)
        normalized_query = question
        if self.retrieval_optimizer and use_query_normalization:
            normalized_query = self.retrieval_optimizer.normalize_and_enhance_query(
                query=question,
                llm_service=self.llm_service,
                conversation_history=None,
                domain_context=self.domain_template.normalization_context
            )
            if normalized_query != question:
                optimization_metadata['transformations'].append('normalization')
                optimization_metadata['normalized_query'] = normalized_query
                logger.info(f"Normalized query: '{question}' → '{normalized_query}'")
        
        # Step 1: Multi-query retrieval (if enabled, uses normalized query)
        if self.retrieval_optimizer and use_multi_query:
            logger.info("Using multi-query RAG fusion")
            retrieved_docs, opt_meta = self.retrieval_optimizer.multi_query_retrieval(
                query=normalized_query,
                vector_store=self.vector_store,
                embeddings_service=self.embeddings_service,
                llm_service=self.llm_service,
                collection_name=self.collection_name,
                num_variations=num_query_variations,
                top_k_per_query=10,
                final_top_k=top_k,
                conversation_history=None,
                metadata_filter=combined_filter if combined_filter else None,
                boost_original=1.5
            )
            optimization_metadata.update(opt_meta)
            optimization_metadata['transformations'].append('multi_query_fusion')
            logger.info(f"Multi-query fusion complete: {len(retrieved_docs)} results")
        
        else:
            # Standard retrieval path
            # Step 1a: Optional query transformation (uses normalized query as base)
            search_query = normalized_query
            
            if self.retrieval_optimizer and (use_query_rewriting or use_hyde):
                if use_hyde:
                    # HyDE: Generate hypothetical answer
                    search_query = self.retrieval_optimizer.hyde_query(
                        normalized_query,
                        self.llm_service,
                        self.system_role
                    )
                    optimization_metadata['transformations'].append('hyde')
                    logger.info("Applied HyDE transformation")
                elif use_query_rewriting:
                    # Query rewriting (on top of normalization)
                    search_query = self.retrieval_optimizer.rewrite_query(
                        normalized_query,
                        self.llm_service
                    )
                    optimization_metadata['transformations'].append('query_rewriting')
                    logger.info(f"Rewrote query: '{normalized_query}' → '{search_query}'")
            
            # Step 2: Generate query embedding (use transformed query)
            query_embedding = self.embeddings_service.embed_text(search_query)
            logger.info("Generated query embedding")
            
            # Step 3: Retrieve relevant documents with metadata filter
            # Retrieve more candidates if using advanced retrieval
            initial_k = top_k * 10 if (self.retrieval_optimizer and (use_hybrid_search or use_reranking)) else top_k
            
            try:
                results = self.vector_store.query(
                    collection_name=self.collection_name,
                    query_embeddings=[query_embedding],
                    n_results=initial_k,
                    metadata_filter=combined_filter if combined_filter else None
                )
            except Exception as e:
                logger.error(f"Error querying vector store: {str(e)}")
                raise
            
            # Extract retrieved documents
            retrieved_docs = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc in enumerate(results['documents'][0]):
                    doc_metadata = results['metadatas'][0][i]
                    
                    # Skip questions, retrieve original content
                    if doc_metadata.get('content_type') == 'question':
                        continue
                    
                    retrieved_docs.append({
                        'text': doc,
                        'metadata': doc_metadata,
                        'distance': results['distances'][0][i],
                        'doc_id': i  # Track original position
                    })
            
            logger.info(f"Retrieved {len(retrieved_docs)} candidates from vector search")
            
            # Step 3.5: Apply distance-based relevance filtering
            if settings.enable_distance_filtering and retrieved_docs:
                before_count = len(retrieved_docs)
                
                # Filter documents by distance threshold
                filtered_docs = [
                    doc for doc in retrieved_docs 
                    if doc['distance'] <= settings.distance_threshold
                ]
                
                # Ensure minimum results (keep best document even if below threshold)
                if len(filtered_docs) < settings.min_results_after_filter and retrieved_docs:
                    # Keep the top documents by distance (lower is better)
                    sorted_docs = sorted(retrieved_docs, key=lambda x: x['distance'])
                    filtered_docs = sorted_docs[:settings.min_results_after_filter]
                    logger.warning(
                        f"Distance filtering would remove all results. "
                        f"Keeping {settings.min_results_after_filter} best document(s) "
                        f"(distance: {filtered_docs[0]['distance']:.3f})"
                    )
                elif filtered_docs:
                    logger.info(
                        f"Distance filtering: {before_count} → {len(filtered_docs)} docs "
                        f"(threshold={settings.distance_threshold}, "
                        f"distances: {['{:.3f}'.format(doc['distance']) for doc in filtered_docs]})"
                    )
                
                retrieved_docs = filtered_docs
            
            # Step 4: Apply advanced retrieval optimization
            if self.retrieval_optimizer and retrieved_docs:
                if use_hybrid_search or use_reranking:
                    optimized_docs, opt_meta = self.retrieval_optimizer.optimize_retrieval(
                        collection_name=self.collection_name,
                        query=question,  # Use ORIGINAL query for reranking
                        vector_results=retrieved_docs,
                        llm_service=self.llm_service,
                        use_hybrid=use_hybrid_search,
                        use_reranking=use_reranking,
                        use_query_rewriting=False,  # Already done above
                        use_hyde=False,  # Already done above
                        top_k_initial=50,
                        top_k_final=top_k
                    )
                    retrieved_docs = optimized_docs
                    optimization_metadata.update(opt_meta)
                    logger.info(f"Applied optimization: {opt_meta['transformations']}")
        
        # Step 5: Retrieve parent chunks if using parent-child strategy
        if retrieved_docs and any(doc.get('metadata', {}).get('has_parent') for doc in retrieved_docs):
            retrieved_docs = self._retrieve_parent_chunks(retrieved_docs)
            logger.info("Replaced child chunks with parent chunks for full context")
        
        logger.info(f"Final document count: {len(retrieved_docs)}")
        
        # Step 6: Calculate confidence score based on retrieval distances
        confidence = self._calculate_confidence(retrieved_docs) if retrieved_docs else 0.0
        
        # Step 7: Generate response using LLM
        try:
            answer = self.llm_service.generate_rag_response(
                query=question,
                retrieved_docs=retrieved_docs,
                system_role=self.system_role
            )
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
        
        logger.info("Generated response")
        
        # Step 8: Detect if response is an "I don't know" type
        is_uncertain = self._detect_uncertainty(answer)
        
        # Step 9: Format sources for citations
        formatted_sources = self._format_sources_for_citations(retrieved_docs) if retrieved_docs else []
        
        # Prepare response
        response = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'is_uncertain': is_uncertain,
            'num_sources': len(retrieved_docs),
            'sources_available': len(retrieved_docs) > 0,
            'optimization_used': optimization_metadata if self.retrieval_optimizer else None
        }
        
        if return_sources:
            response['sources'] = formatted_sources
        
        return response
    
    def _expand_query_with_history(self, query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Expand queries containing pronouns or vague references by incorporating context from recent conversation.
        
        Args:
            query: Current user query
            conversation_history: Previous conversation messages
            
        Returns:
            Expanded query with pronouns/vague terms replaced/supplemented with context
        """
        import re
        
        # Check if query contains pronouns or vague references
        pronouns = ['it', 'that', 'this', 'these', 'those', 'them', 'his', 'her', 'their', 'about']
        vague_terms = ['results', 'details', 'information', 'more', 'other']
        
        query_lower = query.lower()
        
        has_pronoun = any(re.search(rf'\b{pronoun}\b', query_lower) for pronoun in pronouns)
        has_vague_term = any(word in query_lower for word in vague_terms)
        
        logger.info(f"Query expansion check - Query: '{query}', Has pronoun: {has_pronoun}, Has vague term: {has_vague_term}, History length: {len(conversation_history)}")
        
        if (not has_pronoun and not has_vague_term) or not conversation_history:
            return query
        
        # Get the last user message 
        recent_context = []
        for msg in conversation_history[-3:]:  # Last 3 messages for better context
            if msg.get('role') == 'user' and len(msg.get('content', '')) > 0:
                recent_context.append(msg['content'])
        
        logger.info(f"Recent context: {recent_context}")
        
        if not recent_context:
            return query
        
        # Extract key entities from recent conversation
        last_user_query = recent_context[-1]
        
        # Extract key phrases that might be referenced by pronouns/vague terms
        entities = []
        
        # Ordinary Level patterns (check this FIRST before Advanced Level)
        if re.search(r'\b(ordinary level|o-?level|o/l)\b', last_user_query, re.I):
            entities.append('Ordinary Level results')
        # Advanced Level patterns
        elif re.search(r'\b(advanced level|a-?level|a/l)\b', last_user_query, re.I):
            entities.append('Advanced Level results')
        # General education patterns
        elif re.search(r'\b(education|degree|university|school|college)\b', last_user_query, re.I):
            entities.append('education')
        
        # Work patterns  
        if re.search(r'\b(work|job|employment|career)\b', last_user_query, re.I):
            entities.append('work experience')
        
        # Skills patterns
        if re.search(r'\b(skills|technical|competencies)\b', last_user_query, re.I):
            entities.append('skills')
        
        # Volunteer patterns
        if re.search(r'\b(volunteer|extracurricular|organizing)\b', last_user_query, re.I):
            entities.append('volunteering')
        
        # Projects patterns
        if re.search(r'\b(projects|portfolio|built|developed)\b', last_user_query, re.I):
            entities.append('projects')
        
        logger.info(f"Extracted entities: {entities}")
        
        # If we found entities, expand the query
        if entities:
            # Append the most recent entity context to the query
            expanded = f"{query} {entities[0]}"
            logger.info(f"Expanded query: '{query}' → '{expanded}'")
            return expanded
        
        return query
    
    def _retrieve_parent_chunks(self, child_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Replace child chunks with their parent chunks for full context.
        
        Args:
            child_docs: List of retrieved child documents
            
        Returns:
            List of parent documents (deduplicated)
        """
        parent_docs = []
        seen_parents = set()
        
        for doc in child_docs:
            parent_id = doc.get('metadata', {}).get('parent_id')
            if not parent_id or parent_id in seen_parents:
                # No parent or already retrieved, keep original
                if parent_id not in seen_parents:
                    parent_docs.append(doc)
                    if parent_id:
                        seen_parents.add(parent_id)
                continue
            
            # Try to retrieve parent chunk from vector store
            # Search for parent by parent_id in metadata
            try:
                parent_filter = {'parent_id': parent_id, 'chunk_type': 'parent'}
                results = self.vector_store.query(
                    collection_name=self.collection_name,
                    query_embeddings=[doc.get('metadata', {}).get('embedding', [0.0] * 384)],  # Dummy query
                    n_results=1,
                    metadata_filter=parent_filter
                )
                
                if results['documents'] and results['documents'][0]:
                    parent_docs.append({
                        'text': results['documents'][0][0],
                        'metadata': results['metadatas'][0][0],
                        'distance': doc['distance']  # Keep child's relevance score
                    })
                    seen_parents.add(parent_id)
                else:
                    # Parent not found, keep child
                    parent_docs.append(doc)
            except:
                # Error retrieving parent, keep child
                parent_docs.append(doc)
        
        return parent_docs
    
    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieval quality.
        
        Uses distance scores to estimate how confident we are in the answer.
        Lower distance = higher confidence (closer semantic match).
        
        Args:
            documents: List of retrieved documents with distance scores
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not documents:
            return 0.0
        
        # Get average distance of top documents
        distances = [doc.get('distance', 1.0) for doc in documents[:3]]  # Top 3
        avg_distance = sum(distances) / len(distances)
        
        # Convert distance to confidence (inverse relationship)
        # FAISS L2 distance typically ranges 0-2 for good matches
        # Lower distance = higher confidence
        if avg_distance < 0.3:
            confidence = 0.95  # Very high confidence
        elif avg_distance < 0.5:
            confidence = 0.85  # High confidence
        elif avg_distance < 0.8:
            confidence = 0.70  # Moderate confidence
        elif avg_distance < 1.2:
            confidence = 0.50  # Low confidence
        else:
            confidence = 0.30  # Very low confidence
        
        return round(confidence, 2)
    
    def _detect_uncertainty(self, response: str) -> bool:
        """
        Detect if the LLM response indicates uncertainty or inability to answer.
        
        Args:
            response: Generated response text
            
        Returns:
            True if response indicates uncertainty, False otherwise
        """
        uncertainty_phrases = [
            "don't have enough information",
            "i don't know",
            "cannot answer",
            "not sure",
            "don't have that information",
            "insufficient information",
            "unable to answer",
            "can't find",
            "not available",
            "connect you with a human",
            "escalate to",
            "research this further"
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in uncertainty_phrases)
    
    def _format_sources_for_citations(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source documents for easy citation and UI display.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of formatted source dictionaries with citation info
        """
        formatted = []
        
        for idx, doc in enumerate(documents, start=1):
            metadata = doc.get('metadata', {})
            
            # Extract source information
            source_info = {
                'citation_id': idx,
                'citation_label': f"Source {idx}",
                'text': doc.get('text', ''),
                'distance': round(doc.get('distance', 0), 3),
                'relevance': self._distance_to_relevance(doc.get('distance', 1.0)),
                
                # Metadata for UI
                'source_file': metadata.get('filename', metadata.get('source', 'Unknown')),
                'section': metadata.get('section', 'General'),
                'chunk_type': metadata.get('chunk_type', 'standard'),
                
                # For clickable links
                'has_file_path': 'source' in metadata,
                'file_path': metadata.get('source', ''),
                
                # Additional context
                'metadata': metadata
            }
            
            formatted.append(source_info)
        
        return formatted
    
    def _distance_to_relevance(self, distance: float) -> str:
        """
        Convert distance score to human-readable relevance label.
        
        Args:
            distance: FAISS L2 distance score
            
        Returns:
            Relevance label string
        """
        if distance < 0.3:
            return "Highly Relevant"
        elif distance < 0.6:
            return "Relevant"
        elif distance < 1.0:
            return "Moderately Relevant"
        else:
            return "Less Relevant"
    
    def chat(
        self,
        message: str,
        conversation_history: List[Dict[str, str]],
        use_retrieval: bool = True,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        use_query_normalization: bool = True,
        use_query_rewriting: bool = False,
        use_hyde: bool = False,
        use_multi_query: bool = False,
        num_query_variations: int = 3
    ) -> Dict[str, Any]:
        """
        Conversational query with history and advanced retrieval.
        
        Args:
            message: Current user message
            conversation_history: Previous conversation messages (list of dicts with 'role' and 'content')
            use_retrieval: Whether to retrieve context from vector store
            top_k: Number of documents to retrieve if using retrieval
            metadata_filter: Optional metadata filter for retrieval
            use_hybrid_search: Enable hybrid vector+keyword search
            use_reranking: Enable cross-encoder re-ranking
            use_query_normalization: Enable smart query normalization (default: True)
            use_query_rewriting: Enable query rewriting
            use_hyde: Enable HyDE (hypothetical document embeddings)
            use_multi_query: Enable multi-query RAG fusion (more expensive)
            num_query_variations: Number of query variations for multi-query (default: 3)
        
        Returns:
            Dictionary with response and conversation info
        """
        logger.info(f"Processing chat message: {message}")
        logger.info(f"Advanced features: normalize={use_query_normalization}, hybrid={use_hybrid_search}, "
                   f"rerank={use_reranking}, rewrite={use_query_rewriting}, hyde={use_hyde}, multi_query={use_multi_query}")
        
        # Build combined metadata filter
        combined_filter = metadata_filter.copy() if metadata_filter else {}
        top_k = top_k or settings.retrieval_top_k
        
        context = None
        retrieved_docs = None
        optimization_metadata = {'transformations': []}

        # Skip retrieval for greetings / small talk so the assistant simply
        # greets instead of dumping an unrelated document (e.g. a plan).
        if use_retrieval and self._is_smalltalk(message):
            logger.info("Detected greeting/small-talk; skipping retrieval")
            use_retrieval = False

        # Retrieve context if requested
        if use_retrieval:
            try:
                # Step 0: Query normalization (with conversation history for pronoun resolution)
                normalized_query = message
                if self.retrieval_optimizer and use_query_normalization:
                    normalized_query = self.retrieval_optimizer.normalize_and_enhance_query(
                        query=message,
                        llm_service=self.llm_service,
                        conversation_history=conversation_history,
                        domain_context=self.domain_template.normalization_context
                    )
                    if normalized_query != message:
                        optimization_metadata['transformations'].append('normalization')
                        optimization_metadata['normalized_query'] = normalized_query
                        logger.info(f"Normalized query: '{message}' → '{normalized_query}'")
                
                # Step 1: Multi-query retrieval (if enabled, uses normalized query)
                if self.retrieval_optimizer and use_multi_query:
                    logger.info("Using multi-query RAG fusion for chat")
                    retrieved_docs, opt_meta = self.retrieval_optimizer.multi_query_retrieval(
                        query=normalized_query,
                        vector_store=self.vector_store,
                        embeddings_service=self.embeddings_service,
                        llm_service=self.llm_service,
                        collection_name=self.collection_name,
                        num_variations=num_query_variations,
                        top_k_per_query=10,
                        final_top_k=top_k,
                        conversation_history=conversation_history,
                        metadata_filter=combined_filter if combined_filter else None,
                        boost_original=1.5
                    )
                    optimization_metadata.update(opt_meta)
                    optimization_metadata['transformations'].append('multi_query_fusion')
                    logger.info(f"Multi-query fusion complete: {len(retrieved_docs)} results")
                
                else:
                    # Standard retrieval path
                    # Step 1a: Expand query if it contains pronouns (on top of normalization)
                    search_query = self._expand_query_with_history(normalized_query, conversation_history)
                    if search_query != normalized_query:
                        logger.info(f"Expanded query: '{normalized_query}' → '{search_query}'")
                        optimization_metadata['transformations'].append('query_expansion')
                    
                    # Step 2: Optional query transformation
                    if self.retrieval_optimizer and (use_query_rewriting or use_hyde):
                        if use_hyde:
                            # HyDE: Generate hypothetical answer
                            search_query = self.retrieval_optimizer.hyde_query(
                                search_query,
                                self.llm_service,
                                self.system_role
                            )
                            optimization_metadata['transformations'].append('hyde')
                            logger.info("Applied HyDE transformation")
                        elif use_query_rewriting:
                            # Query rewriting (on top of normalization)
                            search_query = self.retrieval_optimizer.rewrite_query(
                                search_query,
                                self.llm_service
                            )
                            optimization_metadata['transformations'].append('query_rewriting')
                            logger.info(f"Rewrote query: '{normalized_query}' → '{search_query}'")
                    
                    # Step 3: Generate query embedding (use transformed query)
                    query_embedding = self.embeddings_service.embed_text(search_query)
                    
                    # Retrieve more candidates if using advanced retrieval
                    initial_k = top_k * 10 if (self.retrieval_optimizer and (use_hybrid_search or use_reranking)) else top_k
                    
                    results = self.vector_store.query(
                        collection_name=self.collection_name,
                        query_embeddings=[query_embedding],
                        n_results=initial_k,
                        metadata_filter=combined_filter if combined_filter else None
                    )
                    
                    # Extract context
                    if results and results.get('documents') and len(results['documents']) > 0 and results['documents'][0]:
                        context = results['documents'][0]
                        retrieved_docs = [
                            {
                                'text': doc,
                                'metadata': results['metadatas'][0][i] if results.get('metadatas') else {},
                                'distance': results['distances'][0][i] if results.get('distances') else 0,
                                'doc_id': i
                            }
                            for i, doc in enumerate(context)
                            if not results['metadatas'][0][i].get('content_type') == 'question'  # Skip questions
                        ]
                        
                        logger.info(f"Retrieved {len(retrieved_docs)} candidates from vector search")
                        
                        # Step 2.5: Apply distance-based relevance filtering
                        if settings.enable_distance_filtering and retrieved_docs:
                            before_count = len(retrieved_docs)
                            
                            # Filter documents by distance threshold
                            filtered_docs = [
                                doc for doc in retrieved_docs 
                                if doc['distance'] <= settings.distance_threshold
                            ]
                            
                            # Ensure minimum results (keep best document even if below threshold)
                            if len(filtered_docs) < settings.min_results_after_filter and retrieved_docs:
                                # Keep the top documents by distance (lower is better)
                                sorted_docs = sorted(retrieved_docs, key=lambda x: x['distance'])
                                filtered_docs = sorted_docs[:settings.min_results_after_filter]
                                logger.warning(
                                    f"Distance filtering would remove all results. "
                                    f"Keeping {settings.min_results_after_filter} best document(s) "
                                    f"(distance: {filtered_docs[0]['distance']:.3f})"
                                )
                            elif filtered_docs:
                                logger.info(
                                    f"Distance filtering: {before_count} → {len(filtered_docs)} docs "
                                    f"(threshold={settings.distance_threshold}, "
                                    f"distances: {['{:.3f}'.format(doc['distance']) for doc in filtered_docs]})"
                                )
                            
                            retrieved_docs = filtered_docs
                        
                        # Step 3: Apply advanced retrieval optimization
                        if self.retrieval_optimizer and retrieved_docs:
                            if use_hybrid_search or use_reranking:
                                optimized_docs, opt_meta = self.retrieval_optimizer.optimize_retrieval(
                                    collection_name=self.collection_name,
                                    query=message,  # Use ORIGINAL query for reranking
                                    vector_results=retrieved_docs,
                                    llm_service=self.llm_service,
                                    use_hybrid=use_hybrid_search,
                                    use_reranking=use_reranking,
                                    use_query_rewriting=False,  # Already done above
                                    use_hyde=False,  # Already done above
                                    top_k_initial=50,
                                    top_k_final=top_k
                                )
                                retrieved_docs = optimized_docs
                                optimization_metadata.update(opt_meta)
                                logger.info(f"Applied optimization: {opt_meta['transformations']}")
                        
                        # Step 4: Retrieve parent chunks if using parent-child
                        if retrieved_docs and any(doc.get('metadata', {}).get('has_parent') for doc in retrieved_docs):
                            retrieved_docs = self._retrieve_parent_chunks(retrieved_docs)
                            logger.info(f"Retrieved parent chunks for full context")
                        
                        context = [doc['text'] for doc in retrieved_docs]
                        logger.info(f"Final document count: {len(retrieved_docs)}")
                    else:
                        logger.info("No documents found in collection for retrieval")
            except Exception as e:
                logger.warning(f"Error during retrieval: {e}. Proceeding without retrieval.")
        
        # Calculate confidence if we have retrieved docs
        confidence = self._calculate_confidence(retrieved_docs) if retrieved_docs else 0.0
        
        # Generate response (limit conversation history to save tokens)
        limited_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        answer = self.llm_service.generate_chat_response(
            query=message,
            conversation_history=limited_history,
            context=context,
            system_role=self.system_role,
            few_shot_examples=self.domain_template.few_shot_examples
        )
        
        # Detect uncertainty in response
        is_uncertain = self._detect_uncertainty(answer)
        
        # Format sources for citations
        formatted_sources = self._format_sources_for_citations(retrieved_docs) if retrieved_docs else []
        
        response = {
            'message': message,
            'answer': answer,
            'confidence': confidence,
            'is_uncertain': is_uncertain,
            'used_retrieval': use_retrieval,
            'sources_available': len(retrieved_docs) > 0 if retrieved_docs else False,
            'optimization_used': optimization_metadata if self.retrieval_optimizer else None
        }
        
        if retrieved_docs:
            response['sources'] = formatted_sources
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            doc_count = self.vector_store.get_collection_count(self.collection_name)
        except:
            doc_count = 0
        
        return {
            'collection_name': self.collection_name,
            'document_count': doc_count,
            'embedding_model': self.embeddings_service.get_model_info(),
            'llm_model': self.llm_service.get_model_info(),
            'system_role': self.system_role
        }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the current collection.
        """
        logger.info(f"Clearing collection: {self.collection_name}")
        self.vector_store.delete_collection(self.collection_name)
        
        # Recreate empty collection immediately
        self.vector_store.create_collection(
            self.collection_name,
            embedding_dimension=settings.embedding_dimension
        )
        self.vector_store.persist()
        logger.info("Collection cleared and recreated")
    
    def load_existing_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Load an existing collection from disk.
        
        Args:
            collection_name: Name of collection to load (uses current if not provided)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        name = collection_name or self.collection_name
        
        try:
            self.vector_store.load_collection(name)
            if collection_name:
                self.collection_name = collection_name
            logger.info(f"Loaded collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection {name}: {str(e)}")
            return False


class MultiClientRAGPipeline:
    """
    Manages multiple RAG pipelines for different clients.
    Each client gets their own collection and optional custom configuration.
    """
    
    def __init__(self):
        """Initialize the multi-client RAG manager."""
        self.pipelines: Dict[str, RAGPipeline] = {}
        logger.info("MultiClientRAGPipeline initialized with lazy loading")
        
        # Don't restore clients on startup - use lazy loading instead
        # Clients will be loaded on-demand when accessed
    
    def create_pipeline(
        self,
        client_id: str,
        system_role: Optional[str] = None,
        domain: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> RAGPipeline:
        """
        Create a new RAG pipeline for a client.

        Args:
            client_id: Unique identifier for the client
            system_role: Custom system role for this client
            domain: Vertical key (telecom/university/generic) for template defaults
            api_key: Optional Groq API key

        Returns:
            RAGPipeline instance for the client
        """
        if client_id in self.pipelines:
            logger.warning(f"Pipeline for client '{client_id}' already exists")
            return self.pipelines[client_id]

        pipeline = RAGPipeline(
            collection_name=f"client_{client_id}",
            api_key=api_key,
            system_role=system_role,
            domain=domain
        )
        
        self.pipelines[client_id] = pipeline
        logger.info(f"Created pipeline for client: {client_id}")
        
        return pipeline
    
    def get_pipeline(self, client_id: str) -> Optional[RAGPipeline]:
        """
        Get a client's RAG pipeline. Lazy-loads from disk if not in memory.
        
        Args:
            client_id: Client identifier
        
        Returns:
            RAGPipeline instance or None if not found
        """
        # If already in memory, return it
        if client_id in self.pipelines:
            return self.pipelines[client_id]

        # Try to lazy-load from disk (client that already has indexed docs)
        if self._load_client_from_disk(client_id):
            return self.pipelines[client_id]

        # Not on disk yet — but it may be a DB client created without documents.
        # Materialize an empty pipeline from its DB config so uploads/chat work.
        if self._create_from_db(client_id):
            return self.pipelines[client_id]

        return None

    def _create_from_db(self, client_id: str) -> bool:
        """Create an empty in-memory pipeline for a DB client with no index yet."""
        try:
            from database import SessionLocal
            from services.client_store import get_client, resolve_persona
            db = SessionLocal()
            try:
                client_row = get_client(db, client_id)
                if client_row is None:
                    return False
                self.create_pipeline(
                    client_id=client_id,
                    system_role=resolve_persona(client_row),
                    domain=client_row.domain,
                )
                logger.info(f"Materialized empty pipeline from DB for client: {client_id}")
                return True
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Could not materialize pipeline from DB for {client_id}: {e}")
            return False
    
    def delete_pipeline(self, client_id: str) -> bool:
        """
        Delete a client's pipeline and data.
        
        Args:
            client_id: Client identifier
        
        Returns:
            True if deleted, False if not found
        """
        if client_id not in self.pipelines:
            return False
        
        # Get collection name
        collection_name = self.pipelines[client_id].collection_name
        
        # Delete the collection (permanently removes files from disk)
        self.pipelines[client_id].vector_store.delete_collection(collection_name)
        
        # Remove from dictionary
        del self.pipelines[client_id]
        
        logger.info(f"Deleted pipeline for client: {client_id} (collection: {collection_name})")
        return True
    
    def list_clients(self) -> List[str]:
        """
        List all client IDs (both loaded and available on disk).
        
        Returns:
            List of client IDs
        """
        from pathlib import Path
        
        # Get clients currently in memory
        loaded_clients = set(self.pipelines.keys())
        
        # Get clients available on disk
        disk_clients = set()
        vector_store_dir = Path(settings.vector_stores_dir) / "faiss"
        if vector_store_dir.exists():
            index_files = vector_store_dir.glob("client_*.index")
            for index_file in index_files:
                collection_name = index_file.stem
                if collection_name.startswith("client_"):
                    client_id = collection_name[7:]  # Remove "client_" prefix
                    disk_clients.add(client_id)
        
        # Combine both sets and return sorted list
        all_clients = loaded_clients.union(disk_clients)
        return sorted(list(all_clients))
    
    def _load_client_from_disk(self, client_id: str) -> bool:
        """
        Lazy-load a single client from disk on-demand.
        
        Args:
            client_id: Client identifier to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        from pathlib import Path
        
        # Check if index file exists
        collection_name = f"client_{client_id}"
        vector_store_dir = Path(settings.vector_stores_dir) / "faiss"
        index_path = vector_store_dir / f"{collection_name}.index"
        
        if not index_path.exists():
            return False
        
        try:
            logger.info(f"Lazy-loading client: {client_id}")

            # Load persona + domain from the DB (source of truth for metadata)
            client_domain = None
            client_persona = None
            try:
                from database import SessionLocal
                from services.client_store import get_client, resolve_persona
                db = SessionLocal()
                try:
                    client_row = get_client(db, client_id)
                    if client_row is not None:
                        client_domain = client_row.domain
                        client_persona = resolve_persona(client_row)
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Could not load client config from DB for {client_id}: {e}")

            # Create pipeline instance
            pipeline = RAGPipeline(
                collection_name=collection_name,
                system_role=client_persona,
                domain=client_domain
            )
            
            # Load the persisted collection
            pipeline.vector_store.load_collection(collection_name)

            # Rebuild BM25 so hybrid search works after a reload (it's in-memory only)
            pipeline.rebuild_bm25_index()

            # Add to pipelines dictionary
            self.pipelines[client_id] = pipeline
            
            logger.info(f"Successfully lazy-loaded client: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to lazy-load client {client_id}: {e}")
            return False
