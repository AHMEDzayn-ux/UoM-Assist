"""
Document Loader Service - Topic-Based Chunking

Implements intelligent topic-based text chunking optimized for 
semantic retrieval in RAG systems.

Features:
- **Topic-based chunking**: Groups semantically related content together
- Lexical cohesion analysis for topic boundary detection
- Keyword extraction and TF-IDF scoring
- Token-based sizing with tiktoken
- Sliding window overlap with sentence boundary preservation
- Rich metadata enrichment
- Special handling for PDFs, FAQs, tables
- Text cleaning pipeline
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import re
import hashlib
import unicodedata
import math
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pypdf import PdfReader
import pdfplumber

from logger import get_logger

logger = get_logger(__name__)


# =========================================================================
# STOPWORDS FOR TOPIC DETECTION
# =========================================================================
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
    'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here',
    'there', 'then', 'once', 'if', 'because', 'until', 'while', 'about',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'over', 'out', 'up', 'down', 'off', 'any', 'its',
    'your', 'their', 'our', 'my', 'his', 'her', 'me', 'him', 'them', 'us',
    'am', 'being', 'etc', 'however', 'therefore', 'thus', 'hence', 'yet',
    'still', 'already', 'even', 'though', 'although', 'whether', 'either',
    'neither', 'much', 'many', 'several', 'get', 'got', 'getting', 'let',
    'lets', 'say', 'said', 'says', 'like', 'make', 'made', 'take', 'took',
}


# =========================================================================
# SEMANTIC METADATA ENRICHMENT - TELECOM DOMAIN
# =========================================================================

METADATA_SYNONYMS = {
    "category": {
        "data": ["internet", "mobile data", "data package", "data plan", "broadband", "connectivity"],
        "social": ["social media", "facebook", "instagram", "whatsapp", "messaging", "chat apps"],
        "entertainment": ["video", "streaming", "youtube", "netflix", "media", "videos"],
        "gaming": ["games", "esports", "online gaming", "multiplayer", "game streaming"],
        "productivity": ["work", "business", "office", "professional", "remote work", "work from home"],
        "combo": ["bundle", "package deal", "combined plan", "all-in-one", "hybrid plan"],
        "roaming": ["international", "global", "travel", "overseas", "abroad", "foreign"]
    },
    "tags": {
        "daily": ["day", "24 hours", "one day", "short term"],
        "weekly": ["week", "7 days", "seven days"],
        "monthly": ["month", "30 days", "long term"],
        "budget": ["cheap", "affordable", "economical", "low cost", "value", "savings"],
        "light_usage": ["light user", "basic", "casual", "minimal usage", "starter"],
        "heavy_usage": ["heavy user", "unlimited", "high usage", "power user", "intensive"],
        "youth": ["student", "young", "teen", "teenagers", "college", "university"],
        "corporate": ["business", "enterprise", "professional", "company", "organization"],
        "travel": ["roaming", "international", "abroad", "overseas", "foreign"],
        "social_media": ["facebook", "instagram", "whatsapp", "twitter", "tiktok", "snapchat"],
        "video_streaming": ["youtube", "netflix", "video", "streaming", "movies", "tv shows"],
        "low_latency": ["fast", "quick", "responsive", "real-time", "instant"],
        "remote_work": ["work from home", "wfh", "telecommute", "home office", "virtual office"]
    },
    "benefits": {
        "unlimited": ["infinite", "no limit", "unrestricted", "limitless", "boundless"],
        "priority": ["fast", "high speed", "premium speed", "prioritized", "accelerated"],
        "night": ["nighttime", "overnight", "off-peak", "evening", "late night"]
    }
}

METADATA_FIELD_LABELS = {
    "package_id": "Package ID",
    "name": "Package Name",
    "category": "Category",
    "validity_days": "Validity",
    "price_lkr": "Price",
    "anytime_data_gb": "Data Allowance",
    "night_data_gb": "Night Data",
    "roaming_data_gb": "Roaming Data",
    "any_network_minutes": "Voice Minutes",
    "sms_count": "SMS Count",
    "tags": "Features",
    "policy_id": "Policy ID",
    "title": "Policy Title"
}


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters."""
    # Token size targets
    target_min_tokens: int = 400
    target_max_tokens: int = 600
    hard_max_tokens: int = 800
    hard_min_tokens: int = 150
    
    # Overlap settings (10-20% of chunk size)
    overlap_tokens: int = 100  # ~80-120 tokens
    
    # Model for tokenization
    encoding_model: str = "cl100k_base"  # GPT-4/ChatGPT tokenizer


@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""
    document_id: str = ""
    section_title: str = ""
    subsection_title: str = ""
    page_number: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    source_type: str = "pdf"
    source_file: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}


@dataclass
class Chunk:
    """Represents a single chunk with text and metadata."""
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to output format."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict()
        }


@dataclass
class TopicSegment:
    """Represents a paragraph/segment with topic analysis."""
    text: str
    keywords: Set[str]
    heading: str = ""
    subheading: str = ""
    segment_type: str = "paragraph"
    topic_id: int = -1  # Assigned after clustering


class TopicAnalyzer:
    """
    Analyzes text for topic coherence and keyword extraction.
    
    Uses TF-IDF-like scoring and lexical cohesion to:
    - Extract meaningful keywords from text
    - Compute similarity between text segments
    - Detect topic boundaries
    """
    
    def __init__(self, min_keyword_length: int = 3, max_keywords: int = 20):
        self.min_keyword_length = min_keyword_length
        self.max_keywords = max_keywords
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.total_documents = 0
    
    def extract_keywords(self, text: str, top_n: int = 15) -> Set[str]:
        """
        Extract keywords from text using term frequency.
        
        Args:
            text: Input text
            top_n: Maximum number of keywords to return
            
        Returns:
            Set of keyword strings
        """
        if not text:
            return set()
        
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stopwords and short words
        words = [w for w in words if w not in STOPWORDS and len(w) >= self.min_keyword_length]
        
        # Count frequencies
        word_counts = Counter(words)
        
        # Get top keywords
        keywords = {word for word, _ in word_counts.most_common(top_n)}
        
        return keywords
    
    def extract_weighted_keywords(self, text: str, corpus: List[str]) -> Dict[str, float]:
        """
        Extract keywords with TF-IDF-like weights.
        
        Args:
            text: Input text
            corpus: List of all document texts for IDF calculation
            
        Returns:
            Dictionary of keyword -> score
        """
        if not text or not corpus:
            return {}
        
        # Calculate document frequencies
        doc_freq = defaultdict(int)
        for doc in corpus:
            words = set(re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower()))
            for word in words:
                if word not in STOPWORDS:
                    doc_freq[word] += 1
        
        # Calculate TF-IDF for current text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOPWORDS and len(w) >= self.min_keyword_length]
        
        word_counts = Counter(words)
        total_words = len(words) or 1
        n_docs = len(corpus) or 1
        
        scores = {}
        for word, count in word_counts.items():
            tf = count / total_words
            idf = math.log(n_docs / (doc_freq.get(word, 0) + 1)) + 1
            scores[word] = tf * idf
        
        return scores
    
    def compute_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """
        Compute Jaccard similarity between two keyword sets.
        
        Args:
            keywords1: First keyword set
            keywords2: Second keyword set
            
        Returns:
            Similarity score between 0 and 1
        """
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_weighted_similarity(self, text1: str, text2: str) -> float:
        """
        Compute weighted similarity using word overlap and position.
        
        More sophisticated than Jaccard - considers:
        - Word frequency in both texts
        - Bi-gram overlap for phrase matching
        - Named entity preservation
        """
        if not text1 or not text2:
            return 0.0
        
        # Extract words
        words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower())) - STOPWORDS
        words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower())) - STOPWORDS
        
        # Word overlap
        word_sim = self.compute_similarity(words1, words2)
        
        # Bi-gram overlap (for phrases)
        bigrams1 = self._extract_bigrams(text1)
        bigrams2 = self._extract_bigrams(text2)
        bigram_sim = self.compute_similarity(bigrams1, bigrams2)
        
        # Named entities (capitalized words)
        entities1 = set(re.findall(r'\b[A-Z][a-z]+\b', text1))
        entities2 = set(re.findall(r'\b[A-Z][a-z]+\b', text2))
        entity_sim = self.compute_similarity(entities1, entities2) if entities1 and entities2 else 0
        
        # Weighted combination
        return 0.5 * word_sim + 0.3 * bigram_sim + 0.2 * entity_sim
    
    def _extract_bigrams(self, text: str) -> Set[str]:
        """Extract word bigrams from text."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        words = [w for w in words if w not in STOPWORDS]
        
        bigrams = set()
        for i in range(len(words) - 1):
            bigrams.add(f"{words[i]}_{words[i+1]}")
        
        return bigrams
    
    def detect_topic_boundaries(
        self,
        segments: List[TopicSegment],
        threshold: float = 0.15
    ) -> List[int]:
        """
        Detect topic boundary positions between segments.
        
        Uses sliding window analysis to find where topic similarity
        drops below threshold, indicating a topic shift.
        
        Args:
            segments: List of TopicSegments with keywords
            threshold: Similarity threshold below which to mark boundary
            
        Returns:
            List of indices where topic boundaries occur
        """
        if len(segments) <= 1:
            return []
        
        boundaries = []
        
        for i in range(1, len(segments)):
            prev_segment = segments[i - 1]
            curr_segment = segments[i]
            
            # Compute similarity
            similarity = self.compute_similarity(
                prev_segment.keywords,
                curr_segment.keywords
            )
            
            # Also check heading changes as strong boundary signal
            heading_changed = (
                prev_segment.heading != curr_segment.heading and 
                curr_segment.heading != ""
            )
            
            # Mark boundary if similarity is low or heading changed
            if similarity < threshold or heading_changed:
                boundaries.append(i)
        
        return boundaries
    
    def cluster_by_topic(
        self,
        segments: List[TopicSegment],
        min_similarity: float = 0.1
    ) -> List[List[int]]:
        """
        Cluster segment indices by topic similarity.
        
        Groups consecutive segments that share topical coherence.
        
        Args:
            segments: List of TopicSegments
            min_similarity: Minimum similarity to keep segments together
            
        Returns:
            List of clusters, each containing segment indices
        """
        if not segments:
            return []
        
        if len(segments) == 1:
            return [[0]]
        
        clusters = []
        current_cluster = [0]
        
        for i in range(1, len(segments)):
            prev_segment = segments[i - 1]
            curr_segment = segments[i]
            
            # Check similarity with cluster (using last segment)
            similarity = self.compute_similarity(
                prev_segment.keywords,
                curr_segment.keywords
            )
            
            # Also check against cluster centroid keywords
            cluster_keywords: Set[str] = set()
            for idx in current_cluster:
                cluster_keywords.update(segments[idx].keywords)
            
            centroid_similarity = self.compute_similarity(
                cluster_keywords,
                curr_segment.keywords
            )
            
            # Use max of adjacent and centroid similarity
            effective_sim = max(similarity, centroid_similarity * 0.8)
            
            # Check for heading-based boundaries
            heading_break = (
                curr_segment.heading != "" and 
                curr_segment.heading != prev_segment.heading
            )
            
            if effective_sim >= min_similarity and not heading_break:
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]
        
        # Don't forget the last cluster
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters


class ProductionChunker:
    """
    Production-grade text chunking for RAG systems.
    
    Implements **TOPIC-BASED CHUNKING** with:
    - Topic detection using lexical cohesion and keyword analysis
    - Groups semantically related paragraphs together
    - Respects natural topic boundaries
    - Token-based sizing with tiktoken
    - Sliding window overlap preserving sentence boundaries
    - Rich metadata enrichment
    """
    
    # Structural patterns for document parsing
    HEADING_PATTERNS = [
        r'^#{1,3}\s+(.+)$',                          # Markdown headings
        r'^([A-Z][A-Za-z\s]+):?\s*$',                # Title case headings
        r'^(\d+\.?\s+[A-Z][A-Za-z\s]+)$',           # Numbered headings
        r'^([A-Z][A-Z\s]+)$',                        # ALL CAPS headings
        r'^(Chapter\s+\d+[:\s].+)$',                 # Chapter headings
        r'^(Section\s+\d+[:\s].+)$',                 # Section headings
    ]
    
    # Section patterns for CV/Resume detection
    SECTION_PATTERNS = {
        "work_experience": [
            r"(?i)^(work\s+experience|employment\s+history|professional\s+experience|career\s+history|experience)",
        ],
        "education": [
            r"(?i)^(education|academic\s+background|qualifications|educational\s+background)",
        ],
        "skills": [
            r"(?i)^(skills|technical\s+skills|competencies|expertise|core\s+competencies)",
        ],
        "volunteer": [
            r"(?i)^(volunteer|volunteering|community\s+service|extracurricular)",
        ],
        "projects": [
            r"(?i)^(projects|key\s+projects|portfolio|personal\s+projects)",
        ],
        "certifications": [
            r"(?i)^(certifications|certificates|licenses|credentials)",
        ],
        "summary": [
            r"(?i)^(summary|profile|objective|about\s+me|professional\s+summary)",
        ],
        "contact": [
            r"(?i)^(contact|contact\s+information|personal\s+details)",
        ],
    }
    
    # FAQ patterns
    FAQ_PATTERNS = [
        r'^Q[:.\s]*(.+?)[\n]+A[:.\s]*(.+?)(?=\n\s*Q[:.\s]|\Z)',  # Q: ... A: ...
        r'^\*\*Q[:.\s]*\*\*(.+?)[\n]+\*\*A[:.\s]*\*\*(.+?)(?=\n\s*\*\*Q|\Z)',  # Bold Q/A
        r'^(?:Question|FAQ)[:.\s]*(.+?)[\n]+(?:Answer)[:.\s]*(.+?)(?=\n\s*(?:Question|FAQ)|\Z)',
    ]
    
    # Table detection patterns
    TABLE_PATTERNS = [
        r'\|.+\|',                                    # Markdown table rows
        r'^\s*[-+]+\s*$',                            # Table separators
    ]
    
    # Code block patterns
    CODE_BLOCK_PATTERN = r'```[\s\S]*?```|`[^`]+`'
    
    # List patterns
    LIST_PATTERNS = [
        r'^\s*[-•*]\s+',                             # Bullet lists
        r'^\s*\d+[.)]\s+',                           # Numbered lists
        r'^\s*[a-z][.)]\s+',                         # Letter lists
    ]
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize the production chunker.
        
        Args:
            config: ChunkConfig with sizing parameters
        """
        self.config = config or ChunkConfig()
        
        # Initialize tokenizer with fallback
        self.encoding = None
        self._use_fallback_tokenizer = False
        
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(self.config.encoding_model)
            logger.info("Using tiktoken for token counting")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken ({e}), using fallback tokenizer")
            self._use_fallback_tokenizer = True
        
        logger.info(f"ProductionChunker initialized with config: "
                   f"target={self.config.target_min_tokens}-{self.config.target_max_tokens} tokens, "
                   f"max={self.config.hard_max_tokens}, overlap={self.config.overlap_tokens}")
    
    # =========================================================================
    # TOKEN UTILITIES
    # =========================================================================
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback."""
        if not text:
            return 0
        
        if self._use_fallback_tokenizer or self.encoding is None:
            # Fallback: estimate ~4 characters per token (GPT-style)
            return len(text) // 4
        
        return len(self.encoding.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens."""
        if self._use_fallback_tokenizer or self.encoding is None:
            # Fallback: estimate ~4 characters per token
            max_chars = max_tokens * 4
            return text[:max_chars]
        
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.encoding.decode(tokens[:max_tokens])
    
    # =========================================================================
    # TEXT CLEANING PIPELINE
    # =========================================================================
    
    def clean_text(self, text: str) -> str:
        """
        Apply comprehensive text cleaning pipeline.
        
        1. Normalize unicode
        2. Fix broken sentences from OCR
        3. Remove extra whitespace
        4. Remove page numbers
        5. Remove table of contents markers
        """
        if not text:
            return ""
        
        # 1. Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Fix common OCR issues
        text = self._fix_ocr_issues(text)
        
        # 3. Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # 4. Remove page numbers
        text = self._remove_page_numbers(text)
        
        # 5. Remove TOC markers
        text = self._remove_toc_markers(text)
        
        return text.strip()
    
    def _fix_ocr_issues(self, text: str) -> str:
        """Fix common OCR artifacts."""
        # Fix broken words (hyphenation at line breaks)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix double spaces
        text = re.sub(r'  +', ' ', text)
        
        # Fix sentences broken across lines
        text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)
        
        # Fix common OCR character substitutions
        replacements = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff',
            ''': "'", ''': "'", '"': '"', '"': '"',
            '–': '-', '—': '-', '…': '...',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure."""
        # Replace multiple blank lines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Clean up spaces around newlines
        text = re.sub(r' *\n *', '\n', text)
        
        return text
    
    def _remove_page_numbers(self, text: str) -> str:
        """Remove standalone page numbers."""
        # Page X, Page X of Y
        text = re.sub(r'\n\s*(?:Page\s*)?\d+\s*(?:of\s*\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Standalone numbers at start/end of lines
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _remove_toc_markers(self, text: str) -> str:
        """Remove table of contents markers."""
        # Remove "Table of Contents" header
        text = re.sub(r'^Table\s+of\s+Contents?\s*\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove TOC entries (text followed by dots and page number)
        text = re.sub(r'^.+\.{3,}\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def remove_headers_footers(self, pages_text: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Remove repeated headers and footers across pages.
        
        Args:
            pages_text: List of (page_text, page_number) tuples
            
        Returns:
            Cleaned list of (page_text, page_number) tuples
        """
        if len(pages_text) < 3:
            return pages_text
        
        # Extract first and last lines from each page
        first_lines = []
        last_lines = []
        
        for text, _ in pages_text:
            lines = text.strip().split('\n')
            if lines:
                first_lines.append(lines[0].strip()[:100])
                last_lines.append(lines[-1].strip()[:100])
        
        # Find repeated headers (appear in >50% of pages)
        header_counts = {}
        for line in first_lines:
            if line:
                header_counts[line] = header_counts.get(line, 0) + 1
        
        footer_counts = {}
        for line in last_lines:
            if line:
                footer_counts[line] = footer_counts.get(line, 0) + 1
        
        threshold = len(pages_text) * 0.5
        repeated_headers = {h for h, c in header_counts.items() if c >= threshold}
        repeated_footers = {f for f, c in footer_counts.items() if c >= threshold}
        
        # Remove repeated headers/footers
        cleaned_pages = []
        for text, page_num in pages_text:
            lines = text.strip().split('\n')
            
            # Remove header if repeated
            if lines and lines[0].strip()[:100] in repeated_headers:
                lines = lines[1:]
            
            # Remove footer if repeated
            if lines and lines[-1].strip()[:100] in repeated_footers:
                lines = lines[:-1]
            
            cleaned_pages.append(('\n'.join(lines), page_num))
        
        return cleaned_pages
    
    # =========================================================================
    # STRUCTURE DETECTION
    # =========================================================================
    
    def detect_heading(self, line: str) -> Optional[str]:
        """Detect if a line is a heading."""
        line = line.strip()
        if not line or len(line) > 100:
            return None
        
        for pattern in self.HEADING_PATTERNS:
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()
        
        return None
    
    def detect_section_type(self, text: str) -> Optional[str]:
        """Detect CV/document section type."""
        first_line = text.strip().split('\n')[0] if text else ""
        
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, first_line.strip()):
                    return section_type
        
        return None
    
    def is_list_item(self, line: str) -> bool:
        """Check if a line is a list item."""
        for pattern in self.LIST_PATTERNS:
            if re.match(pattern, line):
                return True
        return False
    
    def is_table_content(self, text: str) -> bool:
        """Check if text contains table content."""
        for pattern in self.TABLE_PATTERNS:
            if re.search(pattern, text):
                return True
        return False
    
    def extract_faq_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Extract FAQ question-answer pairs."""
        pairs = []
        for pattern in self.FAQ_PATTERNS:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                if len(match) >= 2:
                    pairs.append((match[0].strip(), match[1].strip()))
        return pairs
    
    # =========================================================================
    # STRUCTURAL PARSING
    # =========================================================================
    
    def parse_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse document into structural segments.
        
        Returns list of segments with:
        - type: heading, paragraph, list, table, code, faq
        - content: text content
        - heading: section heading (if applicable)
        - level: heading level (1, 2, 3)
        """
        segments = []
        current_heading = ""
        current_subheading = ""
        
        # Split by double newlines first (paragraphs)
        blocks = re.split(r'\n\s*\n', text)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Check for code blocks
            if re.search(self.CODE_BLOCK_PATTERN, block):
                segments.append({
                    "type": "code",
                    "content": block,
                    "heading": current_heading,
                    "subheading": current_subheading,
                    "preserve": True  # Don't split code blocks
                })
                continue
            
            # Check for tables
            if self.is_table_content(block):
                segments.append({
                    "type": "table",
                    "content": block,
                    "heading": current_heading,
                    "subheading": current_subheading,
                    "preserve": True  # Don't split tables
                })
                continue
            
            # Check for headings
            lines = block.split('\n')
            first_line = lines[0] if lines else ""
            heading = self.detect_heading(first_line)
            
            if heading:
                # Determine heading level
                if first_line.startswith('# ') or first_line.isupper():
                    current_heading = heading
                    current_subheading = ""
                    level = 1
                elif first_line.startswith('## '):
                    current_subheading = heading
                    level = 2
                else:
                    current_subheading = heading
                    level = 2
                
                # If there's content after the heading
                remaining = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
                if remaining:
                    segments.append({
                        "type": "paragraph",
                        "content": remaining,
                        "heading": current_heading,
                        "subheading": current_subheading,
                    })
                continue
            
            # Check for lists
            if any(self.is_list_item(line) for line in lines):
                segments.append({
                    "type": "list",
                    "content": block,
                    "heading": current_heading,
                    "subheading": current_subheading,
                    "preserve": True  # Try to keep lists together
                })
                continue
            
            # Regular paragraph
            segments.append({
                "type": "paragraph",
                "content": block,
                "heading": current_heading,
                "subheading": current_subheading,
            })
        
        return segments
    
    # =========================================================================
    # SENTENCE BOUNDARY HANDLING
    # =========================================================================
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving boundaries."""
        if not text:
            return []
        
        # Common abbreviations that shouldn't end sentences
        abbreviations = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'i.e', 'e.g', 
                        'Inc', 'Ltd', 'Corp', 'Co', 'No', 'Vol', 'Rev', 'Gen', 'Col', 'Fig',
                        'al', 'cf', 'eg', 'ie', 'et'}
        
        # Simple sentence splitting approach
        # Split on sentence-ending punctuation followed by space and capital letter
        sentences = []
        current = []
        words = text.split()
        
        for i, word in enumerate(words):
            current.append(word)
            
            # Check if this word ends a sentence
            if word and word[-1] in '.!?':
                # Check if it's an abbreviation
                word_without_punct = word.rstrip('.!?')
                is_abbrev = word_without_punct in abbreviations
                
                # Check if next word starts with capital (if exists)
                next_starts_capital = (i + 1 < len(words) and 
                                      words[i + 1] and 
                                      words[i + 1][0].isupper())
                
                # End sentence if not abbreviation and next word starts with capital
                if not is_abbrev and (next_starts_capital or i == len(words) - 1):
                    sentences.append(' '.join(current))
                    current = []
        
        # Add remaining words as final sentence
        if current:
            sentences.append(' '.join(current))
        
        return sentences if sentences else [text]
    
    def find_sentence_boundary(self, text: str, target_tokens: int, direction: str = "before") -> int:
        """
        Find the nearest sentence boundary.
        
        Args:
            text: Text to search
            target_tokens: Target token position
            direction: "before" or "after" the target
            
        Returns:
            Character position of the boundary
        """
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return len(text) if direction == "before" else 0
        
        # Build cumulative token counts
        positions = []
        current_pos = 0
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            end_pos = current_pos + len(sentence)
            
            positions.append({
                "start": current_pos,
                "end": end_pos,
                "tokens": current_tokens,
                "end_tokens": current_tokens + sentence_tokens
            })
            
            current_pos = end_pos + 1  # +1 for space
            current_tokens += sentence_tokens
        
        # Find nearest boundary
        if direction == "before":
            for pos in reversed(positions):
                if pos["end_tokens"] <= target_tokens:
                    return pos["end"]
            return positions[0]["end"] if positions else len(text)
        else:  # after
            for pos in positions:
                if pos["tokens"] >= target_tokens:
                    return pos["start"]
            return positions[-1]["start"] if positions else 0
    
    # =========================================================================
    # CHUNKING ENGINE
    # =========================================================================
    
    def merge_small_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge segments that are too small."""
        if not segments:
            return []
        
        merged = []
        buffer = None
        
        for segment in segments:
            tokens = self.count_tokens(segment["content"])
            
            if tokens < self.config.hard_min_tokens and not segment.get("preserve"):
                # Try to merge with buffer
                if buffer:
                    buffer["content"] = buffer["content"] + "\n\n" + segment["content"]
                else:
                    buffer = segment.copy()
            else:
                # Flush buffer if exists
                if buffer:
                    buffer_tokens = self.count_tokens(buffer["content"])
                    if buffer_tokens < self.config.hard_min_tokens:
                        # Merge buffer into current segment
                        segment["content"] = buffer["content"] + "\n\n" + segment["content"]
                    else:
                        merged.append(buffer)
                    buffer = None
                
                merged.append(segment)
        
        # Flush remaining buffer
        if buffer:
            if merged:
                # Merge with last segment
                merged[-1]["content"] = merged[-1]["content"] + "\n\n" + buffer["content"]
            else:
                merged.append(buffer)
        
        return merged
    
    def split_large_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a segment that exceeds the hard maximum token limit."""
        content = segment["content"]
        tokens = self.count_tokens(content)
        
        if tokens <= self.config.hard_max_tokens:
            return [segment]
        
        # If it's a preservable block (table, code), don't split
        if segment.get("preserve"):
            logger.warning(f"Large {segment['type']} block ({tokens} tokens) exceeds max but cannot be split")
            return [segment]
        
        chunks = []
        remaining_text = content
        chunk_idx = 0
        
        while remaining_text and self.count_tokens(remaining_text) > self.config.target_max_tokens:
            # Find split point at sentence boundary
            target_chars = len(remaining_text) * self.config.target_max_tokens // self.count_tokens(remaining_text)
            
            # Find sentence boundary before target
            split_pos = self.find_sentence_boundary(remaining_text, self.config.target_max_tokens, "before")
            
            if split_pos <= 0 or split_pos >= len(remaining_text) - 10:
                # Fallback: split at target position
                split_pos = min(target_chars, len(remaining_text))
            
            chunk_text = remaining_text[:split_pos].strip()
            remaining_text = remaining_text[split_pos:].strip()
            
            new_segment = segment.copy()
            new_segment["content"] = chunk_text
            new_segment["split_index"] = chunk_idx
            chunks.append(new_segment)
            chunk_idx += 1
        
        # Add remaining text
        if remaining_text:
            new_segment = segment.copy()
            new_segment["content"] = remaining_text
            new_segment["split_index"] = chunk_idx
            chunks.append(new_segment)
        
        return chunks
    
    def add_overlap(self, chunks: List[Chunk], overlap_tokens: int) -> List[Chunk]:
        """Add overlapping text between chunks."""
        if len(chunks) <= 1 or overlap_tokens <= 0:
            return chunks
        
        result = []
        
        for i, chunk in enumerate(chunks):
            new_text = chunk.text
            
            # Add overlap from previous chunk
            if i > 0:
                prev_text = chunks[i - 1].text
                prev_sentences = self.split_into_sentences(prev_text)
                
                # Get last N tokens worth of sentences
                overlap_text = ""
                overlap_count = 0
                
                for sentence in reversed(prev_sentences):
                    sentence_tokens = self.count_tokens(sentence)
                    if overlap_count + sentence_tokens <= overlap_tokens:
                        overlap_text = sentence + " " + overlap_text
                        overlap_count += sentence_tokens
                    else:
                        break
                
                if overlap_text:
                    new_text = "[...] " + overlap_text.strip() + " " + new_text
            
            # Create new chunk with updated text
            new_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                text=new_text,
                metadata=ChunkMetadata(
                    **{k: v for k, v in chunk.metadata.to_dict().items()}
                )
            )
            new_chunk.metadata.token_count = self.count_tokens(new_text)
            result.append(new_chunk)
        
        return result
    
    # =========================================================================
    # MAIN CHUNKING METHOD - TOPIC-BASED
    # =========================================================================
    
    def chunk_text(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        source_type: str = "text",
        page_numbers: Optional[Dict[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main chunking method - implements TOPIC-BASED chunking strategy.
        
        Groups semantically related paragraphs into coherent topic chunks.
        
        Args:
            text: Raw text content
            document_id: Unique document identifier
            source_file: Source filename
            source_type: Type of source (pdf, docx, txt)
            page_numbers: Optional mapping of char positions to page numbers
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        logger.info(f"Starting TOPIC-BASED chunking for document: {document_id}")
        
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Check for FAQ format
        faq_pairs = self.extract_faq_pairs(cleaned_text)
        if faq_pairs and len(faq_pairs) >= 3:
            logger.info(f"Detected FAQ format with {len(faq_pairs)} Q&A pairs")
            return self._chunk_faq(faq_pairs, document_id, source_file, source_type)
        
        # Step 3: Parse into paragraphs/segments with structure detection
        segments = self.parse_document_structure(cleaned_text)
        
        # Step 4: Create TopicSegments with keyword extraction
        topic_analyzer = TopicAnalyzer()
        topic_segments: List[TopicSegment] = []
        
        for seg in segments:
            keywords = topic_analyzer.extract_keywords(seg["content"])
            topic_segment = TopicSegment(
                text=seg["content"],
                keywords=keywords,
                heading=seg.get("heading", ""),
                subheading=seg.get("subheading", ""),
                segment_type=seg.get("type", "paragraph")
            )
            topic_segments.append(topic_segment)
        
        logger.info(f"Parsed {len(topic_segments)} segments for topic analysis")
        
        # Step 5: Cluster segments by topic similarity
        topic_clusters = topic_analyzer.cluster_by_topic(
            topic_segments,
            min_similarity=0.1  # Lower = more aggressive grouping
        )
        
        logger.info(f"Identified {len(topic_clusters)} topic clusters")
        
        # Step 6: Create chunks from topic clusters
        raw_chunks = []
        for cluster_idx, cluster in enumerate(topic_clusters):
            # Combine all segments in this topic cluster
            cluster_texts = []
            cluster_heading = ""
            cluster_subheading = ""
            
            for seg_idx in cluster:
                seg = topic_segments[seg_idx]
                cluster_texts.append(seg.text)
                # Use first non-empty heading in cluster
                if not cluster_heading and seg.heading:
                    cluster_heading = seg.heading
                if not cluster_subheading and seg.subheading:
                    cluster_subheading = seg.subheading
            
            combined_text = "\n\n".join(cluster_texts)
            combined_tokens = self.count_tokens(combined_text)
            
            # If topic cluster is too large, split it while keeping topic coherence
            if combined_tokens > self.config.hard_max_tokens:
                sub_chunks = self._split_topic_chunk(
                    combined_text,
                    cluster_heading,
                    cluster_subheading
                )
                raw_chunks.extend(sub_chunks)
            else:
                raw_chunks.append({
                    "content": combined_text,
                    "heading": cluster_heading,
                    "subheading": cluster_subheading,
                    "topic_id": cluster_idx
                })
        
        # Step 7: Merge tiny chunks with neighbors
        merged_chunks = self._merge_small_topic_chunks(raw_chunks)
        
        # Step 8: Create final chunks with metadata
        chunks = []
        for idx, chunk_data in enumerate(merged_chunks):
            content = chunk_data["content"]
            
            # Add section context header
            if chunk_data.get("heading"):
                content = f"[Topic: {chunk_data['heading']}]\n{content}"
            if chunk_data.get("subheading"):
                content = f"[{chunk_data['subheading']}]\n{content}"
            
            chunk_id = self._generate_chunk_id(document_id, idx, content)
            
            metadata = ChunkMetadata(
                document_id=document_id,
                section_title=chunk_data.get("heading", ""),
                subsection_title=chunk_data.get("subheading", ""),
                chunk_index=idx,
                total_chunks=len(merged_chunks),
                token_count=self.count_tokens(content),
                source_type=source_type,
                source_file=source_file
            )
            
            chunk = Chunk(
                chunk_id=chunk_id,
                text=content,
                metadata=metadata
            )
            chunks.append(chunk)
        
        # Step 9: Add overlap between chunks
        if self.config.overlap_tokens > 0:
            chunks = self.add_overlap(chunks, self.config.overlap_tokens)
        
        # Step 10: Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        # Convert to output format
        result = [chunk.to_dict() for chunk in chunks]
        
        # Log statistics
        self._log_chunk_stats(result)
        
        return result
    
    def _split_topic_chunk(
        self,
        text: str,
        heading: str,
        subheading: str
    ) -> List[Dict[str, Any]]:
        """
        Split a large topic chunk while trying to preserve topic coherence.
        
        Uses sentence boundaries and tries to find natural break points.
        """
        sentences = self.split_into_sentences(text)
        
        chunks = []
        current_text = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence exceeds target
            if current_tokens + sentence_tokens > self.config.target_max_tokens and current_text:
                # Save current chunk
                chunks.append({
                    "content": " ".join(current_text),
                    "heading": heading,
                    "subheading": subheading + " (continued)" if chunks else subheading
                })
                current_text = [sentence]
                current_tokens = sentence_tokens
            else:
                current_text.append(sentence)
                current_tokens += sentence_tokens
        
        # Don't forget remainingtext
        if current_text:
            chunks.append({
                "content": " ".join(current_text),
                "heading": heading,
                "subheading": subheading + " (continued)" if len(chunks) > 0 else subheading
            })
        
        return chunks
    
    def _merge_small_topic_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge chunks that are too small with their neighbors.
        
        Prioritizes merging with chunks that share the same topic/heading.
        """
        if not chunks:
            return []
        
        merged = []
        buffer = None
        
        for chunk in chunks:
            tokens = self.count_tokens(chunk["content"])
            
            if tokens < self.config.hard_min_tokens:
                # Try to merge with buffer
                if buffer:
                    buffer_tokens = self.count_tokens(buffer["content"])
                    if buffer_tokens + tokens <= self.config.hard_max_tokens:
                        buffer["content"] = buffer["content"] + "\n\n" + chunk["content"]
                    else:
                        merged.append(buffer)
                        buffer = chunk
                else:
                    buffer = chunk
            else:
                # Flush buffer if exists
                if buffer:
                    buffer_tokens = self.count_tokens(buffer["content"])
                    if buffer_tokens < self.config.hard_min_tokens:
                        # Merge buffer into current chunk if same topic
                        if buffer.get("heading") == chunk.get("heading"):
                            chunk["content"] = buffer["content"] + "\n\n" + chunk["content"]
                        else:
                            merged.append(buffer)
                    else:
                        merged.append(buffer)
                    buffer = None
                
                merged.append(chunk)
        
        # Flush remaining buffer
        if buffer:
            if merged:
                # Try to merge with last chunk
                last_tokens = self.count_tokens(merged[-1]["content"])
                buffer_tokens = self.count_tokens(buffer["content"])
                if last_tokens + buffer_tokens <= self.config.hard_max_tokens:
                    merged[-1]["content"] = merged[-1]["content"] + "\n\n" + buffer["content"]
                else:
                    merged.append(buffer)
            else:
                merged.append(buffer)
        
        return merged
    
    def _chunk_faq(
        self,
        faq_pairs: List[Tuple[str, str]],
        document_id: str,
        source_file: str,
        source_type: str
    ) -> List[Dict[str, Any]]:
        """Handle FAQ format - each Q&A pair becomes one chunk."""
        chunks = []
        
        for idx, (question, answer) in enumerate(faq_pairs):
            content = f"Q: {question}\n\nA: {answer}"
            chunk_id = self._generate_chunk_id(document_id, idx, content)
            
            metadata = ChunkMetadata(
                document_id=document_id,
                section_title="FAQ",
                subsection_title=question[:100],
                chunk_index=idx,
                total_chunks=len(faq_pairs),
                token_count=self.count_tokens(content),
                source_type=source_type,
                source_file=source_file
            )
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": content,
                "metadata": metadata.to_dict()
            })
        
        return chunks
    
    def _generate_chunk_id(self, document_id: str, index: int, content: str) -> str:
        """Generate deterministic chunk ID."""
        unique_string = f"{document_id}_{index}_{content[:100]}"
        hash_value = hashlib.md5(unique_string.encode()).hexdigest()[:12]
        return f"{document_id}_chunk_{index}_{hash_value}"
    
    def _log_chunk_stats(self, chunks: List[Dict[str, Any]]) -> None:
        """Log chunking statistics."""
        if not chunks:
            logger.info("No chunks created")
            return
        
        token_counts = [c["metadata"].get("token_count", 0) for c in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) // len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
        }
        
        logger.info(f"Chunking complete: {stats}")
    
    # =========================================================================
    # HELPER METHOD FOR STATS
    # =========================================================================
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "total_characters": 0
            }
        
        token_counts = [c["metadata"].get("token_count", 0) for c in chunks]
        char_counts = [len(c.get("text", "")) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) // len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_characters": sum(char_counts),
            "avg_characters": sum(char_counts) // len(char_counts)
        }


def _render_markdown_table(rows: List[List[Optional[str]]]) -> str:
    """Render a pdfplumber-extracted table (list of rows of cells) as a GitHub-style
    markdown pipe table, so ``ProductionChunker.is_table_content()`` recognizes it
    and keeps it as one atomic, unsplit chunk."""
    def clean(cell) -> str:
        if cell is None:
            return ""
        return str(cell).replace("|", "\\|").replace("\n", " ").strip()

    cleaned = [[clean(c) for c in row] for row in rows if row]
    if not cleaned:
        return ""
    n_cols = max(len(r) for r in cleaned)
    for r in cleaned:
        r.extend([""] * (n_cols - len(r)))

    header, *data_rows = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * n_cols) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in data_rows)
    return "\n".join(lines)


def _extract_pdf_pages_with_tables(path: Path) -> List[Tuple[str, int]]:
    """Extract each page's body text (tables excluded) plus every table on that
    page rendered as markdown, appended below the text. Uses pdfplumber's
    bounding-box filtering so table cell text isn't duplicated in the body text.
    """
    def _not_within_table_bboxes(bboxes):
        def _in_bbox(obj, bbox):
            v_mid = (obj["top"] + obj["bottom"]) / 2
            h_mid = (obj["x0"] + obj["x1"]) / 2
            x0, top, x1, bottom = bbox
            return (x0 <= h_mid < x1) and (top <= v_mid < bottom)

        def _filter(obj):
            return not any(_in_bbox(obj, bbox) for bbox in bboxes)
        return _filter

    pages: List[Tuple[str, int]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            table_md = [
                _render_markdown_table(t.extract())
                for t in tables
            ]
            table_md = [t for t in table_md if t]

            if tables:
                body_page = page.filter(_not_within_table_bboxes([t.bbox for t in tables]))
                body_text = body_page.extract_text() or ""
            else:
                body_text = page.extract_text() or ""

            parts = [p for p in [body_text.strip(), *table_md] if p]
            page_text = "\n\n".join(parts)
            if page_text.strip():
                pages.append((page_text, page_num))
    return pages


class DocumentLoader:
    """
    Document loader with production-grade chunking.

    Handles PDF loading and intelligent text chunking optimized for RAG retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = 500,  # Target token size (kept for backward compat)
        chunk_overlap: int = 100,  # Overlap in tokens
        config: Optional[ChunkConfig] = None
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Target chunk size in tokens (default: 500)
            chunk_overlap: Overlap between chunks in tokens (default: 100)
            config: Optional ChunkConfig for fine-grained control
        """
        # Create config from parameters if not provided
        if config is None:
            config = ChunkConfig(
                target_min_tokens=max(150, chunk_size - 100),
                target_max_tokens=min(600, chunk_size + 100),
                hard_max_tokens=800,
                hard_min_tokens=150,
                overlap_tokens=chunk_overlap
            )
        
        self.chunker = ProductionChunker(config)
        self.config = config
        
        logger.info(f"DocumentLoader initialized with production chunker")
    
    def load_pdf(self, file_path: str) -> str:
        """
        Load text content from a PDF file, with tables rendered as inline
        markdown so the chunker's table-preservation logic can keep them intact.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content from all pages

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF: {file_path}")

        try:
            pages = _extract_pdf_pages_with_tables(path)
            return "\n\n".join(text for text, _ in pages)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {path.name} ({e}); "
                           f"falling back to pypdf (no table detection)")
            return self._load_pdf_pypdf(path)

    def _load_pdf_pypdf(self, path: Path) -> str:
        """Fallback path for PDFs pdfplumber can't open (e.g. malformed files)."""
        try:
            reader = PdfReader(str(path))
            text_content = []

            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(text)

            return "\n\n".join(text_content)

        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")

    def load_pdf_with_pages(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Load PDF with page number tracking, tables rendered as inline markdown.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of (page_text, page_number) tuples
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF: {file_path}")

        try:
            return _extract_pdf_pages_with_tables(path)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {path.name} ({e}); "
                           f"falling back to pypdf (no table detection)")
            return self._load_pdf_with_pages_pypdf(path)

    def _load_pdf_with_pages_pypdf(self, path: Path) -> List[Tuple[str, int]]:
        """Fallback path for PDFs pdfplumber can't open (e.g. malformed files)."""
        try:
            reader = PdfReader(str(path))
            pages = []

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append((text, page_num))

            return pages

        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text content to split
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Extract document info from metadata
        metadata = metadata or {}
        document_id = metadata.get("document_id", metadata.get("filename", "unknown"))
        source_file = metadata.get("source", metadata.get("filename", ""))
        source_type = metadata.get("source_type", "text")
        
        # Use production chunker
        chunks = self.chunker.chunk_text(
            text=text,
            document_id=document_id,
            source_file=source_file,
            source_type=source_type
        )
        
        # Merge additional metadata
        for chunk in chunks:
            chunk["metadata"] = {**chunk.get("metadata", {}), **metadata}
            # Add backward-compatible fields
            chunk["chunk_index"] = chunk["metadata"].get("chunk_index", 0)
            chunk["chunk_size"] = len(chunk.get("text", ""))
        
        return chunks
    
    def load_and_chunk_pdf(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load a PDF and split it into chunks in one operation.
        
        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks with metadata
        """
        path = Path(file_path)
        
        # Build metadata
        file_metadata = {
            "source": str(path),
            "filename": path.name,
            "document_id": path.stem,
            "source_type": "pdf",
            **(metadata or {})
        }
        
        # Load with page tracking
        pages = self.load_pdf_with_pages(file_path)
        
        # Remove repeated headers/footers
        pages = self.chunker.remove_headers_footers(pages)
        
        # Combine pages
        full_text = "\n\n".join([text for text, _ in pages])
        
        # Chunk the text
        chunks = self.chunk_text(full_text, file_metadata)
        
        return chunks
    
    # =========================================================================
    # JSON LOADING AND CHUNKING (PRODUCTION-LEVEL)
    # =========================================================================
    
    def load_json(self, file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load and parse JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data (dict or list)
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If file is not valid JSON
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        if path.suffix.lower() != '.json':
            raise ValueError(f"File must be JSON: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {str(e)}")
    
    def detect_json_structure(self, data: Union[Dict, List]) -> Dict[str, Any]:
        """
        Analyze JSON structure to determine chunking strategy.
        
        Args:
            data: Parsed JSON data
            
        Returns:
            Dictionary with structure information
        """
        structure_info = {
            "type": "unknown",
            "array_field": None,
            "total_items": 0,
            "sample_keys": [],
            "suggested_text_fields": [],
            "suggested_metadata_fields": []
        }
        
        if isinstance(data, list):
            structure_info["type"] = "array"
            structure_info["total_items"] = len(data)
            
            if data and isinstance(data[0], dict):
                structure_info["sample_keys"] = list(data[0].keys())
                structure_info["suggested_text_fields"], structure_info["suggested_metadata_fields"] = \
                    self._suggest_field_mapping(data[0])
        
        elif isinstance(data, dict):
            # Check if it's a wrapper with array field
            for key, value in data.items():
                if isinstance(value, list) and value:
                    structure_info["type"] = "object_with_array"
                    structure_info["array_field"] = key
                    structure_info["total_items"] = len(value)
                    
                    if isinstance(value[0], dict):
                        structure_info["sample_keys"] = list(value[0].keys())
                        structure_info["suggested_text_fields"], structure_info["suggested_metadata_fields"] = \
                            self._suggest_field_mapping(value[0])
                    break
            
            if structure_info["type"] == "unknown":
                # Single object
                structure_info["type"] = "single_object"
                structure_info["total_items"] = 1
                structure_info["sample_keys"] = list(data.keys())
                structure_info["suggested_text_fields"], structure_info["suggested_metadata_fields"] = \
                    self._suggest_field_mapping(data)
        
        return structure_info
    
    def _suggest_field_mapping(self, sample_obj: Dict) -> Tuple[List[str], List[str]]:
        """
        Suggest which fields should be used for text vs metadata.
        
        Args:
            sample_obj: Sample JSON object
            
        Returns:
            Tuple of (text_fields, metadata_fields)
        """
        text_fields = []
        metadata_fields = []
        
        # Common text field names (case-insensitive)
        text_field_patterns = [
            'description', 'content', 'text', 'body', 'message', 'details',
            'answer', 'response', 'summary', 'info', 'information',
            'question', 'query', 'title', 'name', 'subject',
            'rules', 'policy', 'procedure', 'steps', 'instructions',  # Policy-specific fields
            'benefits', 'features', 'includes'  # Package-specific fields
        ]
        
        # Common metadata field names
        metadata_field_patterns = [
            'id', 'type', 'category', 'tag', 'status', 'date', 'created',
            'updated', 'author', 'priority', 'version', 'code', 'sku'
        ]
        
        for field, value in sample_obj.items():
            field_lower = field.lower()
            
            # Check if it's a text field based on name
            is_text_field = any(pattern in field_lower for pattern in text_field_patterns)
            is_metadata_field = any(pattern in field_lower for pattern in metadata_field_patterns)
            
            # Prioritize field name patterns over type
            if is_text_field:
                text_fields.append(field)
            elif is_metadata_field:
                metadata_fields.append(field)
            # Then check value type
            elif isinstance(value, str):
                if len(value) > 50:  # Long strings likely contain text
                    text_fields.append(field)
                else:
                    metadata_fields.append(field)
            elif isinstance(value, (int, float, bool)):
                metadata_fields.append(field)
            elif isinstance(value, (dict, list)):
                metadata_fields.append(field)  # Complex types as metadata by default
            else:
                metadata_fields.append(field)
        
        return text_fields, metadata_fields
    
    def json_object_to_text(
        self,
        obj: Dict[str, Any],
        text_fields: Optional[List[str]] = None,
        include_all_fields: bool = True,
        max_nesting_depth: int = 3
    ) -> str:
        """
        Convert JSON object to human-readable text.
        
        Args:
            obj: JSON object to convert
            text_fields: Specific fields to prioritize as main text
            include_all_fields: Whether to include all fields or just text_fields
            max_nesting_depth: Maximum depth for nested object expansion
            
        Returns:
            Formatted text representation
        """
        lines = []
        
        # Prioritize text fields first
        if text_fields:
            for field in text_fields:
                if field in obj:
                    value = obj[field]
                    if isinstance(value, str) and value.strip():
                        lines.append(value.strip())
                    else:
                        # For non-string text fields (lists, dicts), format them
                        formatted_value = self._format_json_value(value, depth=0, max_depth=max_nesting_depth)
                        if formatted_value:
                            field_name = field.replace('_', ' ').title()
                            lines.append(f"{field_name}: {formatted_value}")
        
        # Add other fields if requested
        if include_all_fields:
            for field, value in obj.items():
                if text_fields and field in text_fields:
                    continue  # Already added
                
                formatted_value = self._format_json_value(value, depth=0, max_depth=max_nesting_depth)
                if formatted_value:
                    field_name = field.replace('_', ' ').title()
                    lines.append(f"{field_name}: {formatted_value}")
        
        return "\n".join(lines)
    
    def _format_json_value(self, value: Any, depth: int = 0, max_depth: int = 3) -> str:
        """Recursively format JSON value as text."""
        if depth >= max_depth:
            return str(value)
        
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, list):
            if not value:
                return ""
            # If list of primitives, join them
            if all(isinstance(v, (str, int, float, bool)) for v in value):
                return ", ".join(str(v) for v in value)
            # If list of objects, format each
            return " | ".join(self._format_json_value(v, depth+1, max_depth) for v in value[:5])  # Limit to 5 items
        elif isinstance(value, dict):
            parts = []
            for k, v in list(value.items())[:10]:  # Limit to 10 fields
                formatted = self._format_json_value(v, depth+1, max_depth)
                if formatted:
                    k_clean = k.replace('_', ' ').title()
                    parts.append(f"{k_clean}: {formatted}")
            return " • ".join(parts)
        elif value is None:
            return ""
        else:
            return str(value)
    
    def _enrich_text_with_metadata(
        self,
        text: str,
        metadata: Dict[str, Any],
        enable_semantic: bool = True
    ) -> str:
        """Enrich text with semantic metadata for better retrieval.
        
        Adds structured metadata as semantic prefix to make metadata values
        searchable. For example, a package with {"category": "data", "tags": ["budget"]}
        becomes searchable for queries like "cheap internet plan".
        
        Args:
            text: Original chunk text
            metadata: Metadata dictionary to enrich
            enable_semantic: Whether to add synonym expansion
            
        Returns:
            Enriched text with semantic metadata prefix
        """
        if not metadata:
            return text
        
        enrichment_parts = []
        
        # Package name (most important)
        if "name" in metadata:
            enrichment_parts.append(f"📦 {metadata['name']}")
        
        # Category with synonyms
        if "category" in metadata:
            category = metadata["category"]
            category_text = f"Category: {category}"
            if enable_semantic and category in METADATA_SYNONYMS.get("category", {}):
                synonyms = METADATA_SYNONYMS["category"][category]
                category_text += f" ({', '.join(synonyms[:3])})"  # Add top 3 synonyms
            enrichment_parts.append(category_text)
        
        # Price information
        if "price_lkr" in metadata:
            price = metadata["price_lkr"]
            enrichment_parts.append(f"💰 Price: LKR {price}")
            # Add price tier semantic tags
            if price < 500:
                enrichment_parts.append("[Budget-friendly] [Affordable] [Economical]")
            elif price < 1500:
                enrichment_parts.append("[Mid-range] [Value pack]")
            else:
                enrichment_parts.append("[Premium] [High-end] [Professional]")
        
        # Validity period
        if "validity_days" in metadata:
            days = metadata["validity_days"]
            if days == 1:
                enrichment_parts.append("⏱️ Validity: Daily (1 day, 24 hours, short-term)")
            elif days == 7:
                enrichment_parts.append("⏱️ Validity: Weekly (7 days, one week)")
            elif days == 30:
                enrichment_parts.append("⏱️ Validity: Monthly (30 days, one month, long-term)")
            else:
                enrichment_parts.append(f"⏱️ Validity: {days} days")
        
        # Tags with synonyms
        if "tags" in metadata:
            tags = metadata["tags"] if isinstance(metadata["tags"], list) else [metadata["tags"]]
            tag_texts = []
            for tag in tags:
                tag_text = tag.replace('_', ' ').title()
                if enable_semantic and tag in METADATA_SYNONYMS.get("tags", {}):
                    synonyms = METADATA_SYNONYMS["tags"][tag]
                    tag_text += f" ({', '.join(synonyms[:2])})"  # Add top 2 synonyms
                tag_texts.append(tag_text)
            enrichment_parts.append(f"🏷️ Features: {' | '.join(tag_texts)}")
        
        # Data benefits (nested extraction)
        if "benefits" in metadata and isinstance(metadata["benefits"], dict):
            benefits = metadata["benefits"]
            benefit_texts = []
            
            if benefits.get("anytime_data_gb"):
                data_gb = benefits["anytime_data_gb"]
                benefit_texts.append(f"{data_gb}GB data allowance")
            
            if benefits.get("night_data_gb"):
                night_gb = benefits["night_data_gb"]
                benefit_texts.append(f"{night_gb}GB bonus night data")
            
            if benefits.get("any_network_minutes"):
                mins = benefits["any_network_minutes"]
                benefit_texts.append(f"{mins} voice minutes")
            
            if benefits.get("sms_count"):
                sms = benefits["sms_count"]
                benefit_texts.append(f"{sms} SMS")
            
            # Boolean benefits
            if benefits.get("whatsapp_unlimited"):
                benefit_texts.append("Unlimited WhatsApp (messaging, chat, no limit)")
            if benefits.get("facebook_unlimited"):
                benefit_texts.append("Unlimited Facebook (social media, no limit)")
            if benefits.get("instagram_unlimited"):
                benefit_texts.append("Unlimited Instagram (social media, no limit)")
            if benefits.get("youtube_unlimited"):
                benefit_texts.append("Unlimited YouTube (video streaming, no limit)")
            if benefits.get("gaming_priority_network"):
                benefit_texts.append("Gaming priority network (low latency, fast, responsive)")
            if benefits.get("zoom_priority"):
                benefit_texts.append("Zoom priority (video conferencing, meetings)")
            if benefits.get("teams_priority"):
                benefit_texts.append("Microsoft Teams priority (collaboration, work)")
            
            if benefit_texts:
                enrichment_parts.append(f"✨ Benefits: {' • '.join(benefit_texts)}")
        
        # Regions for roaming packages
        if "regions_supported" in metadata:
            regions = metadata["regions_supported"]
            if isinstance(regions, list):
                enrichment_parts.append(f"🌍 Coverage: {', '.join(regions)} (international, travel, abroad)")
        
        # Activation methods
        if "activation" in metadata and isinstance(metadata["activation"], dict):
            activation = metadata["activation"]
            methods = []
            if activation.get("ussd"):
                methods.append(f"USSD: {activation['ussd']}")
            if activation.get("sms"):
                methods.append(f"SMS: {activation['sms']}")
            if activation.get("app"):
                methods.append("Mobile App")
            if methods:
                enrichment_parts.append(f"📱 Activate via: {' | '.join(methods)}")
        
        # Eligibility
        if "eligibility" in metadata and isinstance(metadata["eligibility"], dict):
            eligibility = metadata["eligibility"]
            elig_parts = []
            if eligibility.get("prepaid"):
                elig_parts.append("Prepaid")
            if eligibility.get("postpaid"):
                elig_parts.append("Postpaid")
            if eligibility.get("age_maximum"):
                elig_parts.append(f"Youth (under {eligibility['age_maximum']})")
            if elig_parts:
                enrichment_parts.append(f"👥 Eligible for: {', '.join(elig_parts)}")
        
        # Policy-specific fields
        if "policy_id" in metadata:
            enrichment_parts.append(f"📋 Policy: {metadata['policy_id']}")
        
        if "title" in metadata and "name" not in metadata:  # For policies
            enrichment_parts.append(f"📄 {metadata['title']}")
        
        # Combine enrichment with original text
        if enrichment_parts:
            enrichment_header = "\n".join(enrichment_parts)
            return f"{enrichment_header}\n\n{'='*60}\n\n{text}"
        
        return text
    
    def chunk_json_objects(
        self,
        json_objects: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        text_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        group_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Convert JSON objects to chunks.
        
        Args:
            json_objects: List of JSON objects to chunk
            metadata: Base metadata to attach to all chunks
            text_fields: Fields to use as main text content
            metadata_fields: Fields to extract as metadata
            group_size: Number of objects to group per chunk (1 = one object per chunk)
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        base_metadata = metadata or {}
        
        # Auto-detect fields if not provided
        if json_objects and not (text_fields or metadata_fields):
            if isinstance(json_objects[0], dict):
                text_fields, metadata_fields = self._suggest_field_mapping(json_objects[0])
                logger.info(f"Auto-detected text fields: {text_fields}")
                logger.info(f"Auto-detected metadata fields: {metadata_fields}")
        
        # Group objects if needed
        for i in range(0, len(json_objects), group_size):
            group = json_objects[i:i+group_size]
            
            # Convert to text
            text_parts = []
            for obj in group:
                text = self.json_object_to_text(obj, text_fields=text_fields)
                if text:
                    text_parts.append(text)
            
            if not text_parts:
                continue
            
            # Combine text
            combined_text = "\n\n".join(text_parts)
            
            # Extract metadata from first object in group
            obj_metadata = {}
            if metadata_fields and group:
                for field in metadata_fields:
                    if field in group[0]:
                        obj_metadata[field] = group[0][field]
            
            # Also extract commonly used nested fields for enrichment
            if group:
                first_obj = group[0]
                # Keep the original object fields for semantic enrichment
                for key in ["name", "category", "price_lkr", "validity_days", "tags", 
                           "benefits", "regions_supported", "activation", "eligibility",
                           "policy_id", "title"]:
                    if key in first_obj and key not in obj_metadata:
                        obj_metadata[key] = first_obj[key]
            
            # ======== SEMANTIC METADATA ENRICHMENT ========
            # Enrich text with semantic metadata for better retrieval
            enriched_text = self._enrich_text_with_metadata(
                combined_text,
                obj_metadata,
                enable_semantic=True
            )
            
            # Store original text in metadata for display
            obj_metadata["original_text"] = combined_text
            
            # Create chunk metadata
            chunk_metadata = {
                **base_metadata,
                **obj_metadata,
                "chunk_index": len(chunks),
                "source_type": "json",
                "objects_in_chunk": len(group)
            }
            
            # Use standard chunking if text is too large
            if len(enriched_text) > 2000:  # Increased threshold for enriched text
                sub_chunks = self.chunk_text(enriched_text, chunk_metadata)
                chunks.extend(sub_chunks)
            else:
                # Single chunk with enriched text
                chunks.append({
                    "text": enriched_text,  # Enriched for embeddings
                    "metadata": chunk_metadata,
                    "chunk_id": f"json_chunk_{len(chunks)}",
                    "chunk_size": len(enriched_text)
                })
        
        return chunks
    
    def load_and_chunk_json(
        self,
        file_path: str,
        array_field: Optional[str] = None,
        text_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        group_size: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load JSON file and convert to chunks - Production ready.
        
        Perfect for customer care use cases:
        - Package information: Each package = 1 chunk
        - FAQ databases: Each Q&A pair = 1 chunk
        - Product catalogs: Each product = 1 chunk
        - Support tickets: Each ticket = 1 chunk
        
        Args:
            file_path: Path to JSON file
            array_field: Field containing array of objects (auto-detected if None)
            text_fields: Fields to use as text content (auto-detected if None)
            metadata_fields: Fields to use as metadata (auto-detected if None)
            group_size: Objects per chunk (1 recommended for customer care)
            metadata: Additional metadata to attach
            
        Returns:
            List of chunks ready for vector store
            
        Example JSON structures:
        
        1. Package Tracking:
        ```json
        {
          "packages": [
            {
              "tracking_id": "PKG123",
              "status": "In Transit",
              "description": "Express delivery package containing electronics",
              "destination": "New York, NY",
              "estimated_delivery": "2024-01-15"
            }
          ]
        }
        ```
        
        2. FAQ Database:
        ```json
        {
          "faqs": [
            {
              "question": "How do I track my package?",
              "answer": "You can track your package by entering your tracking number...",
              "category": "Shipping",
              "tags": ["tracking", "delivery"]
            }
          ]
        }
        ```
        """
        path = Path(file_path)
        
        # Build base metadata
        file_metadata = {
            "source": str(path),
            "filename": path.name,
            "document_id": path.stem,
            "source_type": "json",
            **(metadata or {})
        }
        
        # Load JSON
        data = self.load_json(file_path)
        
        # Detect structure
        structure = self.detect_json_structure(data)
        logger.info(f"JSON structure detected: {structure['type']}, {structure['total_items']} items")
        
        # Extract array of objects
        json_objects = []
        if structure["type"] == "array":
            json_objects = data
        elif structure["type"] == "object_with_array":
            field = array_field or structure["array_field"]
            if field and field in data:
                json_objects = data[field]
            else:
                raise ValueError(f"Array field '{field}' not found in JSON")
        elif structure["type"] == "single_object":
            json_objects = [data]
        
        # Use auto-detected fields if not provided
        if not text_fields:
            text_fields = structure["suggested_text_fields"]
        if not metadata_fields:
            metadata_fields = structure["suggested_metadata_fields"]
        
        # Convert to chunks
        chunks = self.chunk_json_objects(
            json_objects=json_objects,
            metadata=file_metadata,
            text_fields=text_fields,
            metadata_fields=metadata_fields,
            group_size=group_size
        )
        
        logger.info(f"Created {len(chunks)} chunks from {len(json_objects)} JSON objects")
        return chunks
    
    def load_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Universal document loader - auto-detects file type.
        
        Supports:
        - PDF files (.pdf)
        - JSON files (.json)
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata
            **kwargs: Format-specific arguments (e.g., array_field for JSON)
            
        Returns:
            List of chunks
            
        Raises:
            ValueError: If file type not supported
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            logger.info(f"Loading PDF: {path.name}")
            return self.load_and_chunk_pdf(file_path, metadata)
        
        elif suffix == '.json':
            logger.info(f"Loading JSON: {path.name}")
            return self.load_and_chunk_json(file_path, metadata=metadata, **kwargs)
        
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .json")
    
    # =========================================================================
    # CHUNK STATISTICS
    # =========================================================================
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        return self.chunker.get_chunk_stats(chunks)
    
    # =========================================================================
    # BACKWARD COMPATIBILITY METHODS
    # =========================================================================
    
    def chunk_with_parent_child(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Parent-Child Chunking Strategy (backward compatible).
        
        Creates:
        - Parent chunks (large): Full context for LLM
        - Child chunks (small): Precise search matching
        
        Args:
            text: Text content to split
            metadata: Optional metadata
            
        Returns:
            Dictionary with parent_chunks and child_chunks lists
        """
        if not text or not text.strip():
            return {"parent_chunks": [], "child_chunks": []}
        
        metadata = metadata or {}
        document_id = metadata.get("document_id", "unknown")
        
        # Create parent chunks (larger)
        parent_config = ChunkConfig(
            target_min_tokens=800,
            target_max_tokens=1200,
            hard_max_tokens=1500,
            hard_min_tokens=400,
            overlap_tokens=150
        )
        parent_chunker = ProductionChunker(parent_config)
        
        parent_chunks_raw = parent_chunker.chunk_text(
            text=text,
            document_id=document_id,
            source_file=metadata.get("source", ""),
            source_type=metadata.get("source_type", "text")
        )
        
        # Create child chunks (smaller) - use default chunker
        child_chunks_raw = self.chunk_text(text, metadata)
        
        # Format for backward compatibility
        parent_chunks = []
        for i, chunk in enumerate(parent_chunks_raw):
            parent_id = f"parent_{i}"
            parent_chunks.append({
                "text": chunk["text"],
                "chunk_id": parent_id,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "parent_id": parent_id,
                    "chunk_type": "parent",
                    "parent_index": i
                }
            })
        
        child_chunks = []
        for i, chunk in enumerate(child_chunks_raw):
            # Assign to nearest parent
            parent_idx = min(i * len(parent_chunks_raw) // max(len(child_chunks_raw), 1), 
                           len(parent_chunks_raw) - 1)
            parent_id = f"parent_{parent_idx}"
            child_id = f"{parent_id}_child_{i}"
            
            child_chunks.append({
                "text": chunk["text"],
                "chunk_id": child_id,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "child_index": i
                }
            })
        
        return {
            "parent_chunks": parent_chunks,
            "child_chunks": child_chunks
        }
    
    def generate_qa_pairs(
        self,
        chunk_text: str,
        num_questions: int = 3,
        llm_service: Optional[Any] = None
    ) -> List[str]:
        """
        Generate hypothetical questions for a chunk.
        
        Args:
            chunk_text: Text content
            num_questions: Number of questions to generate
            llm_service: LLMService instance for generation
            
        Returns:
            List of generated questions
        """
        if not llm_service:
            return self._generate_simple_questions(chunk_text, num_questions)
        
        try:
            prompt = f"""Based on the following text, generate {num_questions} specific questions that this text can answer. 
Each question should be concise and directly answerable by the text.
Format: One question per line, no numbering.

Text:
{chunk_text[:1000]}

Questions:"""
            
            response = llm_service.generate_response(
                query=prompt,
                system_prompt="You are a helpful assistant that generates relevant questions from documentation."
            )
            
            questions = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('#')]
            return questions[:num_questions]
        
        except Exception:
            return self._generate_simple_questions(chunk_text, num_questions)
    
    def _generate_simple_questions(self, text: str, num: int = 3) -> List[str]:
        """Generate simple keyword-based questions as fallback."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        questions = []
        
        templates = [
            "What is {}?",
            "How does {} work?",
            "Tell me about {}",
            "Explain {}",
            "What are the details of {}?"
        ]
        
        for i, sentence in enumerate(sentences[:num]):
            words = sentence.split()[:5]
            if len(words) >= 2:
                topic = ' '.join(words)
                template = templates[i % len(templates)]
                questions.append(template.format(topic))
        
        return questions[:num]
    
    def chunk_with_qa_generation(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        llm_service: Optional[Any] = None,
        generate_qa: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk text and optionally generate QA pairs for each chunk.
        
        Args:
            text: Text content
            metadata: Optional metadata
            llm_service: LLMService for QA generation
            generate_qa: Whether to generate QA pairs
            
        Returns:
            List of chunks with optional generated questions
        """
        chunks = self.chunk_text(text, metadata)
        
        if generate_qa and chunks:
            for chunk in chunks:
                chunk["generated_questions"] = self.generate_qa_pairs(
                    chunk["text"],
                    num_questions=2,
                    llm_service=llm_service
                )
        
        return chunks
