"""
LYRA CLEAN - CONTEXT INJECTION SERVICE
======================================

Zero-latency semantic context injection via SQLite queries.

Replaces:
- lyra_core/enhanced_proxy_hook.py (RAM DataFrame scans)
- lyra_core/graph_contextual_injection.py (NetworkX traversals)

Key Improvements:
- O(log N) neighbor queries (was O(N) DataFrame scan)
- Async/await non-blocking I/O
- Simple keyword extraction (TF-IDF, no heavy NLP)
- Direct SQLite queries (no in-memory graph)

Author: Refactored from Lyra_Uni_3 legacy
"""
from __future__ import annotations

import re
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import Counter

from database.engine import ISpaceDB
from core.physics.bezier import PhysicsState


# ============================================================================
# KEYWORD EXTRACTION (Lightweight, no bert/nltk)
# ============================================================================

# Stop words (minimal set for English/French)
STOP_WORDS = {
    # English
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'can', 'could', 'should', 'may', 'might', 'must', 'that', 'this',
    'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
    # French
    'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'dans',
    'sur', 'par', 'pour', 'avec', 'de', 'du', 'au', 'aux', 'ce', 'cet',
    'cette', 'ces', 'qui', 'que', 'dont', 'où'
}


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    Extract keywords from text using simple TF-IDF-like scoring.

    Strategy:
    - Tokenize by word boundaries
    - Filter stop words
    - Keep words length >= 4
    - Score by frequency × length (longer words = more specific)

    Args:
        text: Input text (user prompt)
        top_k: Number of keywords to return

    Returns:
        List of keywords, sorted by relevance

    Performance: O(n) where n = text length

    Example:
        >>> extract_keywords("What is entropy in information theory?")
        ['entropy', 'information', 'theory']
    """
    # Tokenize (alphanumeric + accents)
    words = re.findall(r"[a-zA-Z\u00C0-\u00FF]{4,}", text.lower())

    # Filter stop words
    words = [w for w in words if w not in STOP_WORDS]

    if not words:
        return []

    # Score by frequency × word length (longer = more specific)
    scores = Counter()
    for word in words:
        scores[word] += len(word)

    # Return top K
    return [word for word, _ in scores.most_common(top_k)]


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

@dataclass(frozen=True)
class GraphContext:
    """
    Immutable context extracted from semantic graph.

    Attributes:
        query_keywords: Keywords extracted from user prompt
        neighbor_concepts: Semantic neighbors with weights
        total_weight: Sum of neighbor weights (contextual strength)
        latency_ms: Time to extract context
    """
    query_keywords: List[str]
    neighbor_concepts: List[Dict[str, Any]]  # [{target, weight, kappa}, ...]
    total_weight: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            'query_keywords': self.query_keywords,
            'neighbor_concepts': self.neighbor_concepts,
            'total_weight': self.total_weight,
            'latency_ms': self.latency_ms
        }


class ContextInjector:
    """
    Semantic context injector using SQLite graph queries.

    Workflow:
    1. Extract keywords from user prompt
    2. Query SQLite for neighbors (single optimized query)
    3. Format context according to physics state (δr)
    4. Return enriched prompt

    Performance Target: < 50ms per injection
    """

    def __init__(self, db: ISpaceDB):
        """
        Initialize injector.

        Args:
            db: Database engine instance
        """
        self.db = db

    async def extract_context(
        self,
        user_prompt: str,
        max_keywords: int = 5,
        max_neighbors: int = 15,
        min_weight: float = 0.1
    ) -> GraphContext:
        """
        Extract semantic context from prompt.

        Args:
            user_prompt: User input text
            max_keywords: Maximum keywords to extract
            max_neighbors: Maximum semantic neighbors
            min_weight: Minimum PPMI weight threshold

        Returns:
            GraphContext with neighbors and metadata

        Performance: O(k × log N) where k = keywords, N = graph size
        """
        start_time = time.time()

        # Step 1: Extract keywords
        keywords = extract_keywords(user_prompt, top_k=max_keywords)

        if not keywords:
            # No keywords → empty context
            return GraphContext(
                query_keywords=[],
                neighbor_concepts=[],
                total_weight=0.0,
                latency_ms=(time.time() - start_time) * 1000
            )

        # Step 2: Query neighbors (batch query for all keywords)
        neighbors = await self.db.get_multi_neighbors(
            keywords,
            limit=max_neighbors
        )

        # Filter by weight
        neighbors = [n for n in neighbors if n['weight'] >= min_weight]

        # Compute total contextual weight
        total_weight = sum(n['weight'] for n in neighbors)

        latency_ms = (time.time() - start_time) * 1000

        return GraphContext(
            query_keywords=keywords,
            neighbor_concepts=neighbors,
            total_weight=total_weight,
            latency_ms=latency_ms
        )

    def format_context(
        self,
        context: GraphContext,
        physics_state: PhysicsState,
        max_length: int = 200
    ) -> str:
        """
        Format graph context as text block.

        Args:
            context: Extracted graph context
            physics_state: Current physics state (for δr scheduling)
            max_length: Maximum context string length

        Returns:
            Formatted context string

        Format:
            [SEMANTIC CONTEXT]
            Keywords: entropy, information, theory
            Related: chaos (0.85), order (0.72), randomness (0.68)
        """
        if not context.neighbor_concepts:
            return ""

        # Format neighbor list
        neighbors_str = ", ".join([
            f"{n['target']} ({n['weight']:.2f})"
            for n in context.neighbor_concepts[:5]  # Top 5
        ])

        # Build context block
        context_text = (
            f"[SEMANTIC CONTEXT]\n"
            f"Keywords: {', '.join(context.query_keywords)}\n"
            f"Related: {neighbors_str}"
        )

        # Truncate if too long
        if len(context_text) > max_length:
            context_text = context_text[:max_length] + "..."

        return context_text

    async def build_enriched_prompt(
        self,
        user_prompt: str,
        system_prompt: str,
        physics_state: PhysicsState,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        enable_context: bool = True
    ) -> Dict[str, Any]:
        """
        Build complete enriched prompt for LLM.

        Args:
            user_prompt: User input
            system_prompt: System instructions
            physics_state: Current physics state (δr for scheduling)
            conversation_history: Previous messages [{role, content}, ...]
            enable_context: Whether to inject semantic context

        Returns:
            Dict with keys:
            - messages: List of {role, content} for Ollama API
            - context: GraphContext metadata (for logging)
            - latency_ms: Total build time

        Scheduling (δr):
        - δr > 0.5: Context AFTER user prompt (delayed injection)
        - δr < -0.5: Context BEFORE user prompt (pre-loading)
        - δr ≈ 0: Context interleaved naturally
        """
        start_time = time.time()

        # Extract semantic context
        graph_context = None
        context_text = ""

        if enable_context:
            graph_context = await self.extract_context(user_prompt)
            context_text = self.format_context(graph_context, physics_state)

        # Build messages array
        messages = []

        # System message (always first)
        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # Conversation history (if provided)
        if conversation_history:
            messages.extend(conversation_history)

        # User prompt + context (scheduling based on δr)
        delta_r = physics_state.delta_r

        if delta_r >= 0.5:
            # Context AFTER user prompt (delayed)
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            if context_text:
                messages.append({
                    "role": "system",
                    "content": f"[DELAYED CONTEXT]\n{context_text}"
                })

        elif delta_r <= -0.5:
            # Context BEFORE user prompt (pre-loaded)
            if context_text:
                messages.append({
                    "role": "system",
                    "content": f"[PRELOADED CONTEXT]\n{context_text}"
                })
            messages.append({
                "role": "user",
                "content": user_prompt
            })

        else:
            # Context interleaved (natural)
            user_content = user_prompt
            if context_text:
                user_content = f"{user_prompt}\n\n{context_text}"

            messages.append({
                "role": "user",
                "content": user_content
            })

        latency_ms = (time.time() - start_time) * 1000

        return {
            "messages": messages,
            "context": graph_context.to_dict() if graph_context else None,
            "latency_ms": latency_ms
        }


# ============================================================================
# CONVERSATION MEMORY (Session history)
# ============================================================================

class ConversationMemory:
    """
    Manages conversation history with sliding window.

    Replaces:
    - lyra_core/memory_store.py (RAM-only cache)

    Features:
    - Persistent storage via SQLite events table
    - Sliding window (last N messages)
    - Token budget estimation
    """

    def __init__(self, db: ISpaceDB):
        """
        Initialize memory manager.

        Args:
            db: Database engine
        """
        self.db = db

    async def get_recent_messages(
        self,
        session_id: str,
        max_messages: int = 20,
        max_tokens: int = 4000
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation messages with token budget.

        Args:
            session_id: Session UUID
            max_messages: Maximum messages to retrieve
            max_tokens: Approximate token budget (4 chars ≈ 1 token)

        Returns:
            List of {role, content} dicts, oldest first

        Strategy:
        - Fetch last N messages from database
        - Trim oldest messages if token budget exceeded
        - Always keep system prompt if present
        """
        messages = await self.db.get_conversation_messages(session_id, limit=max_messages)

        # Estimate tokens (rough: 4 chars ≈ 1 token)
        def estimate_tokens(msgs: List[Dict]) -> int:
            return sum(len(m.get('content', '')) for m in msgs) // 4

        # Trim if over budget (keep newest)
        while len(messages) > 1 and estimate_tokens(messages) > max_tokens:
            # Remove oldest message (but keep system if first)
            if messages[0].get('role') == 'system':
                messages.pop(1)  # Remove second message
            else:
                messages.pop(0)  # Remove first message

        return messages


# ============================================================================
# UTILITIES
# ============================================================================

def build_system_prompt(physics_state: PhysicsState) -> str:
    """
    Build dynamic system prompt based on physics state.

    Args:
        physics_state: Current physics parameters

    Returns:
        System prompt string

    Incorporates:
    - κ (kappa): Style hints (formal vs exploratory)
    - ρ (rho): Tone guidance (expansive vs conservative)
    """
    from core.physics.bezier import map_kappa_to_style_hints

    base_prompt = "You are Lyra, an AI assistant with adaptive behavior."

    # Style hints from kappa
    style_hints = map_kappa_to_style_hints(physics_state.kappa)
    if style_hints:
        style_section = "\n\nStyle Guidelines:\n" + "\n".join(f"- {hint}" for hint in style_hints)
        base_prompt += style_section

    # Tone from rho
    rho = physics_state.rho
    if rho > 0.3:
        base_prompt += "\n\nTone: Exploratory and expansive. Feel free to introduce new concepts."
    elif rho < -0.3:
        base_prompt += "\n\nTone: Conservative and precise. Stick to established knowledge."

    return base_prompt
