"""
LLM Service for RAG System
Handles text generation using Groq API with LangChain
"""

import logging
from typing import List, Dict, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    """
    Service for generating responses using LLM.
    Uses Groq API with LangChain for text generation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the LLM Service.
        
        Args:
            api_key: Groq API key (defaults to settings)
            model_name: Model to use (defaults to settings)
            temperature: Sampling temperature (defaults to settings)
            max_tokens: Maximum tokens in response (defaults to settings)
        """
        self.provider = (settings.llm_provider or "groq").lower()
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        if self.provider == "gemini":
            self.api_key = api_key or settings.google_api_key
            self.model_name = model_name or settings.gemini_model
            self.utility_model_name = settings.gemini_utility_model
        else:
            self.api_key = api_key or settings.groq_api_key
            self.model_name = model_name or settings.llm_model
            self.utility_model_name = settings.llm_utility_model

        self.llm = self._build_client(self.model_name, self.temperature, self.max_tokens)
        # Lightweight client for cheap auxiliary calls + resilient fallback
        self.utility_llm = self._build_client(self.utility_model_name, 0.0, 512)

        logger.info(
            f"LLMService initialized: provider={self.provider}, model={self.model_name}, "
            f"utility={self.utility_model_name}"
        )

    def _build_client(self, model: str, temperature: float, max_tokens: int):
        """Create a chat client for the configured provider (Groq or Gemini)."""
        if not self.api_key:
            logger.warning(f"No API key for provider '{self.provider}'. LLM disabled.")
            return None
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self.api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        return ChatGroq(
            api_key=self.api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    def generate_response(
        self,
        query: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        lightweight: bool = False
    ) -> str:
        """
        Generate a response to a user query.

        Args:
            query: User's question or prompt
            context: List of relevant context passages (from RAG retrieval)
            system_prompt: Custom system prompt (optional)
            conversation_history: Previous conversation messages (optional)
            lightweight: Use the cheap utility model (for internal preprocessing
                         like query normalization) instead of the main model

        Returns:
            Generated response text
        """
        # Build system prompt
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        # Build messages
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Build user message with context
        user_message = self._build_user_message(query, context)
        messages.append(HumanMessage(content=user_message))

        logger.info(f"Generating response for query: {query[:100]}...")

        try:
            # Pick the model: cheap utility model for internal preprocessing,
            # main model for actual answers.
            llm = self.utility_llm if (lightweight and self.utility_llm) else self.llm
            response = llm.invoke(messages)
            response_text = response.content
            # Gemini may return content as a list of parts — normalize to string.
            if isinstance(response_text, list):
                response_text = "".join(
                    p if isinstance(p, str) else (p.get("text") or "") if isinstance(p, dict) else ""
                    for p in response_text
                )
            
            logger.info(f"Generated response: {response_text[:100]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def generate_rag_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        system_role: Optional[str] = None
    ) -> str:
        """
        Generate a RAG response using retrieved documents.
        
        Args:
            query: User's question
            retrieved_docs: List of retrieved documents with 'text' and 'metadata'
            system_role: Role description for the assistant (e.g., "university advisor")
        
        Returns:
            Generated response text
        """
        # Extract context from retrieved documents
        context = [doc.get("text", "") for doc in retrieved_docs]
        
        # Build system prompt for RAG
        system_prompt = self._get_rag_system_prompt(system_role)
        
        response = self.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt
        )
        
        # Clean citation phrases from response
        return self._clean_citation_phrases(response)
    
    def generate_chat_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        context: Optional[List[str]] = None,
        system_role: Optional[str] = None,
        few_shot_examples: Optional[List[str]] = None
    ) -> str:
        """
        Generate a conversational response with history.

        Args:
            query: Current user message
            conversation_history: Previous conversation messages
            context: Optional context from RAG retrieval
            system_role: Client persona (e.g. telecom support, university advisor)
            few_shot_examples: Domain-specific example dialogues to inject

        Returns:
            Generated response text
        """
        system_prompt = self._get_chat_system_prompt(
            role=system_role,
            few_shot_examples=few_shot_examples
        )
        
        response = self.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt,
            conversation_history=conversation_history
        )
        
        # Clean citation phrases from response
        return self._clean_citation_phrases(response)
    
    def detect_emotion(self, message: str) -> Dict[str, Any]:
        """Classify the customer's emotional state (cheap utility-model call).

        Returns {emotion, intensity (1-5), wants_human}.
        """
        default = {"emotion": "neutral", "intensity": 1, "wants_human": False}
        if not self.utility_llm or not (message or "").strip():
            return default
        system = (
            "You are an emotion classifier for customer-support messages. "
            "Classify ONLY the customer's message. Respond with ONLY compact JSON, no prose: "
            '{"emotion":"angry|frustrated|confused|neutral|happy","intensity":1,"wants_human":false}. '
            "intensity is 1-5 (strength of the emotion). wants_human is true only if they explicitly "
            "ask for a human/agent/manager or to escalate."
        )
        try:
            import json
            import re
            resp = self.utility_llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=message)]
            )
            text = resp.content
            if isinstance(text, list):
                text = "".join(
                    p if isinstance(p, str) else (p.get("text") or "") if isinstance(p, dict) else ""
                    for p in text
                )
            m = re.search(r"\{.*\}", text, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            emotion = str(data.get("emotion", "neutral")).lower().strip()
            if emotion not in {"angry", "frustrated", "confused", "neutral", "happy"}:
                emotion = "neutral"
            try:
                intensity = max(1, min(5, int(data.get("intensity", 1))))
            except (TypeError, ValueError):
                intensity = 1
            return {
                "emotion": emotion,
                "intensity": intensity,
                "wants_human": bool(data.get("wants_human", False)),
            }
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return default

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context. If you don't know the answer, say so."""
    
    def _get_rag_system_prompt(self, role: Optional[str] = None) -> str:
        """
        Get system prompt for RAG-based responses with strict grounding.
        
        Args:
            role: Optional role description (e.g., "assistant")
        """
        role_desc = role or "helpful assistant"
        
        return f"""You are a {role_desc}.

🌐 LANGUAGE (decide from the user's QUESTION, in this order):
1. If the question contains ANY Sinhala script (සිංහල) → reply ENTIRELY in Sinhala script.
2. Else if it contains ANY Tamil script (தமிழ்) → reply ENTIRELY in Tamil script.
3. Else if it is romanized Sinhala / "Singlish" (Sinhala words in Latin letters, e.g.
   "mata plan eka gana danaganna oney") → reply in natural, fluent Sinhala SCRIPT.
4. Else if it is romanized Tamil (Tamil words in Latin letters) → reply in natural,
   fluent Tamil SCRIPT.
5. Else (the question is in plain English) → reply in ENGLISH.
• Match the user's language exactly; never switch languages on the user.
• The retrieved information is usually in English — when replying in Sinhala or Tamil,
  translate its MEANING into that language; never paste it verbatim.

📋 CONTENT RULES:
• Answer naturally, as if you know it personally
• Always give complete answers
• Only omit information if truly irrelevant

✨ FORMATTING RULES (VERY IMPORTANT):
• Use bullet points (•) for lists - NOT numbered lists unless specific order matters
• Break long content into SHORT paragraphs (2-3 sentences max)
• Add line breaks between different topics/sections
• Make it visually appealing and easy to read at a glance
• Do NOT use emojis (they get read aloud in voice mode) — use plain text

Examples:

❌ BAD (poor formatting):
"He has worked on several projects including a database with fast retrieval and ACID operations and a 4-bit Nano Processor using VHDL and Basys 3 Board and an indoor sports court booking system with SMS alerts and a disaster management platform and an e-commerce platform."

✅ GOOD (well formatted):
"He has worked on several projects:

• Database with fast retrieval and ACID-compliant operations
• 4-bit Nano Processor using VHDL and Basys 3 Board
• Indoor sports court booking system with SMS alerts
• Disaster management platform with real-time reporting
• E-commerce platform with optimized database design

Each project showcases his skills in database design and system development."

Always format responses for easy scanning and readability!"""
    
    def _get_chat_system_prompt(
        self,
        role: Optional[str] = None,
        few_shot_examples: Optional[List[str]] = None
    ) -> str:
        """
        Build the conversational system prompt from the client persona and
        domain-specific few-shot examples.

        Args:
            role: Persona text for this client's vertical
            few_shot_examples: Domain example dialogues (injected, not baked in)
        """
        role_desc = role or "a helpful assistant"

        if few_shot_examples:
            examples_block = "\n\n".join(few_shot_examples)
        else:
            # Domain-neutral fallback so no vertical's examples bleed into another
            examples_block = (
                'User: "tell me about the premium option"\n'
                'You: "The premium option includes [key benefits]. Want the full details?"\n'
                'User: "what does it cost"\n'
                'You: "It\'s [price] — I can break down what\'s included if you\'d like."'
            )

        return f"""You are {role_desc}, having a natural conversation.

LANGUAGE (decide from the user's LAST message, in this order):
1. If it contains ANY Sinhala script (සිංහල) → reply ENTIRELY in Sinhala script.
2. Else if it contains ANY Tamil script (தமிழ்) → reply ENTIRELY in Tamil script.
3. Else if it is romanized Sinhala / "Singlish" (Sinhala words in Latin letters) → reply
   in natural, fluent Sinhala SCRIPT — never romanized.
4. Else if it is romanized Tamil (Tamil words in Latin letters) → reply in natural,
   fluent Tamil SCRIPT — never romanized.
5. Else (plain English) → reply in ENGLISH.
• Match their language every turn; never switch languages on the user.
• Knowledge-base content is usually in English — when replying in Sinhala or Tamil, convey
  its MEANING in that language rather than pasting it verbatim.

IDENTITY & ACCOUNT (critical):
• You do NOT have access to the user's account, phone number, current plan, balance, or usage. NEVER say "your plan is…", "your current plan", or claim to know their account or personal situation.
• Present knowledge-base information as GENERAL options ("We offer…", "The Senior 55+ plan is…"), never as the user's personal data.
• For account-specific questions (my bill, my plan, my usage), tell them to check their official account portal or offer to connect them to a human agent.

FOCUS (most important):
• Answer ONLY the exact question asked — never volunteer unrelated products, plans, prices, or details.
• If the retrieved context is not relevant to the question, IGNORE it. Never dump a product/plan spec sheet unless the user asked about that product.
• Don't repeat what you already said. For greetings or small talk, reply briefly and ask how you can help — do NOT pitch.

STYLE:
• Read the conversation history; follow-ups like 'yes' or 'what about that' refer to the CURRENT topic. Stay on topic.
• Be confident and direct — no 'might/could be/possibly', and never show your reasoning.
• Lead with the 2-3 key points, then stop and offer more ("Want the details?").
• Answer ONLY from the knowledge base; if it's not there, say so and offer to connect them to a human.
• Use short paragraphs and bullet points (•) for lists. Keep it easy to scan.

Example of good, focused answering:
{examples_block}"""
    
    def _build_user_message(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Build the user message with optional context.
        
        Args:
            query: User's question
            context: Optional list of context passages
        
        Returns:
            Formatted user message
        """
        if not context:
            return query
        
        # Build message with context - NO LABELS to avoid citations
        context_str = "\n\n---\n\n".join(context)
        
        message = f"""Information:

{context_str}

---

Question: {query}

IMPORTANT Instructions:
1. Use ALL the information provided above to give a COMPLETE  answer
2. Do NOT hide important details - include everything relevant
4. Answer naturally
5. FORMAT properly: Use bullet points for lists, short paragraphs, and line breaks - NO huge text blocks!

Answer:"""
        
        return message
    
    def _clean_citation_phrases(self, response: str) -> str:
        """
        Remove citation phrases that mention context sources.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Cleaned response without citation phrases
        """
        import re
        
        # Patterns to remove (case-insensitive)
        patterns = [
            r"According to \[Context \d+\],?\s*",
            r"Based on \[Context \d+\],?\s*",
            r"As mentioned in \[Context \d+\],?\s*",
            r"From \[Context \d+\],?\s*",
            r"In \[Context \d+\],?\s*",
            r"\[Context \d+\] states that\s*",
            r"\[Context \d+\] mentions that\s*",
            r"\[Context \d+\] indicates that\s*",
            r"\[Source \d+\],?\s*",
            r"According to the (provided )?context,?\s*",
            r"Based on the (provided )?context,?\s*",
            r"From the (provided )?context,?\s*",
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Clean up double spaces and leading spaces
        cleaned = re.sub(r"  +", " ", cleaned)
        cleaned = cleaned.strip()
        
        # Capitalize first letter if needed
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        logger.debug(f"Cleaned citations: '{response[:100]}...' → '{cleaned[:100]}...'")
        return cleaned
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM configuration.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "provider": "Groq"
        }
    
    def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.llm.invoke([HumanMessage(content="Hello")])
            logger.info("Groq API connection test successful")
            return True
        except Exception as e:
            logger.error(f"Groq API connection test failed: {str(e)}")
            return False
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate tokens for
        
        Returns:
            Estimated token count (rough approximation)
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def validate_input_size(self, messages: List[str]) -> bool:
        """
        Validate that input doesn't exceed token limits.
        
        Args:
            messages: List of message texts to validate
        
        Returns:
            True if within limits, False otherwise
        """
        total_text = " ".join(messages)
        estimated_tokens = self.estimate_tokens(total_text)
        
        # Groq models typically have 8k-32k context windows
        # We'll use a conservative limit
        max_input_tokens = 7000
        
        if estimated_tokens > max_input_tokens:
            logger.warning(
                f"Input size ({estimated_tokens} tokens) exceeds limit ({max_input_tokens})"
            )
            return False
        
        return True
