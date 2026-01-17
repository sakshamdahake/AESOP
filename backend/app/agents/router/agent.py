"""
Router Agent - Smart query classification with multiple signals.
Uses pattern matching + LLM for nuanced routing decisions.
"""

import json
import re
from typing import Optional, Tuple

from app.schemas.session import RouterDecision, SessionContext
from app.agents.router.prompts import (
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_TEMPLATE,
    format_previous_context,
)
from app.embeddings.bedrock import embed_query, cosine_similarity
from app.logging import logger


# Deictic and reference patterns that indicate context-dependent queries
CONTEXT_REFERENCE_PATTERNS = [
    # Deictic references
    r"\bthese\s+(studies|papers|results|findings|trials|articles)\b",
    r"\bthose\s+(studies|papers|results|findings|trials|articles)\b",
    r"\bthe\s+(studies|papers|results|findings|trials|articles)\b",
    r"\babove\s+(studies|papers|results|findings|mentioned)\b",
    r"\bprevious(ly)?\s+(mentioned|discussed|found|shown)\b",
    
    # Explicit references
    r"\b(first|second|third|1st|2nd|3rd)\s+(study|paper|article|finding)\b",
    r"\bpaper\s*[#]?\d+\b",
    r"\bpmid\s*[:#]?\s*\d+\b",
    r"\bstudy\s*[#]?\d+\b",
    
    # Clarification patterns
    r"\bcan you (explain|clarify|elaborate|expand)\b",
    r"\bwhat (do|did|does) (you|it|they|that) mean\b",
    r"\btell me more about\b",
    r"\bmore (details|information|info) (on|about)\b",
    r"\bwhat about the\b",
    r"\bhow (do|does|did) (this|that|these|those)\b",
    
    # Comparative references to prior results
    r"\bcompare (these|those|the|them)\b",
    r"\bwhich (of these|of those|one|study|paper)\b",
    r"\bbetween (these|those|the)\b",
    r"\bsummarize (the|these|those)\b",
]

# Patterns indicating a new/different topic
NEW_TOPIC_PATTERNS = [
    r"\bwhat (is|are|causes?|treatments?)\s+\w+\b",  # New "what is X" questions
    r"\bhow (do|does|is|are)\s+\w+\s+(work|caused|treated|diagnosed)\b",
    r"\btell me about\s+\w+\b",  # Generic "tell me about X"
]


class RouterAgent:
    """
    Smart query classifier using multiple signals:
    1. Pattern matching for deictic/reference markers
    2. Keyword overlap with original query
    3. Embedding similarity (as secondary signal)
    4. LLM classification for ambiguous cases
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self._reference_patterns = [re.compile(p, re.IGNORECASE) for p in CONTEXT_REFERENCE_PATTERNS]
        self._new_topic_patterns = [re.compile(p, re.IGNORECASE) for p in NEW_TOPIC_PATTERNS]
    
    def route(
        self,
        current_query: str,
        session_context: Optional[SessionContext],
    ) -> RouterDecision:
        """
        Classify query using multiple signals.
        """
        # Fast path: New session
        if session_context is None:
            logger.info("ROUTER_FAST_PATH route=full_graph reason=new_session")
            return RouterDecision(
                route="full_graph",
                reasoning="New session with no previous context",
                similarity_score=0.0,
                is_new_session=True,
            )
        
        # Fast path: Session exists but has no original query (empty session)
        if not session_context.original_query:
            logger.info("ROUTER_FAST_PATH route=full_graph reason=empty_session")
            return RouterDecision(
                route="full_graph",
                reasoning="Session exists but has no previous query context",
                similarity_score=0.0,
                is_new_session=True,
            )
        
        # Signal 1: Check for context reference patterns
        has_context_reference, reference_type = self._detect_context_references(current_query)
        
        # Signal 2: Check keyword overlap with original query
        keyword_overlap = self._compute_keyword_overlap(
            current_query, 
            session_context.original_query
        )
        
        # Signal 3: Embedding similarity (secondary signal)
        current_embedding = embed_query(current_query)
        
        # Handle empty session embedding (new session without full orchestration)
        if session_context.query_embedding and len(session_context.query_embedding) > 0:
            embedding_similarity = cosine_similarity(
                current_embedding,
                session_context.query_embedding,
            )
        else:
            # No stored embedding, default to 0 (will rely on other signals)
            embedding_similarity = 0.0
            logger.debug("ROUTER_EMPTY_EMBEDDING: Session has no stored embedding, using 0.0")
        
        # Signal 4: Check for new topic indicators
        has_new_topic_pattern = self._detect_new_topic(current_query)
        
        logger.info(
            "ROUTER_SIGNALS",
            extra={
                "has_context_reference": has_context_reference,
                "reference_type": reference_type,
                "keyword_overlap": round(keyword_overlap, 3),
                "embedding_similarity": round(embedding_similarity, 3),
                "has_new_topic_pattern": has_new_topic_pattern,
                "current_query": current_query[:60],
                "original_query": session_context.original_query[:60],
            },
        )
        
        # Decision logic using multiple signals
        decision = self._make_decision(
            has_context_reference=has_context_reference,
            reference_type=reference_type,
            keyword_overlap=keyword_overlap,
            embedding_similarity=embedding_similarity,
            has_new_topic_pattern=has_new_topic_pattern,
            current_query=current_query,
            session_context=session_context,
        )
        
        return decision
    
    def _detect_context_references(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detect deictic markers and explicit references to prior context.
        Returns (has_reference, reference_type).
        """
        query_lower = query.lower()
        
        for pattern in self._reference_patterns:
            if pattern.search(query_lower):
                # Categorize the reference type
                if any(word in query_lower for word in ["explain", "clarify", "elaborate", "more details"]):
                    return True, "clarification"
                elif any(word in query_lower for word in ["compare", "between", "which"]):
                    return True, "comparison"
                elif any(word in query_lower for word in ["these", "those", "the studies", "the papers"]):
                    return True, "deictic"
                elif any(word in query_lower for word in ["first", "second", "paper #", "pmid"]):
                    return True, "explicit_reference"
                else:
                    return True, "general_reference"
        
        return False, None
    
    def _detect_new_topic(self, query: str) -> bool:
        """Detect patterns that suggest a completely new topic."""
        for pattern in self._new_topic_patterns:
            if pattern.search(query):
                return True
        return False
    
    def _compute_keyword_overlap(self, query1: str, query2: str) -> float:
        """
        Compute keyword overlap between two queries.
        Focuses on medical/scientific terms, ignores stop words.
        """
        stop_words = {
            "what", "are", "is", "the", "a", "an", "of", "for", "in", "on", 
            "to", "with", "and", "or", "how", "does", "do", "did", "can", 
            "could", "would", "should", "these", "those", "this", "that",
            "about", "from", "by", "be", "been", "being", "have", "has",
            "had", "there", "their", "they", "them", "it", "its", "my",
            "your", "our", "me", "you", "we", "i", "he", "she", "who",
            "which", "when", "where", "why", "if", "then", "so", "but",
            "not", "no", "yes", "all", "any", "some", "more", "most",
            "other", "into", "over", "such", "only", "same", "than",
            "very", "just", "also", "now", "here", "well", "way", "may",
            "use", "used", "using", "tell", "show", "find", "found",
        }
        
        def extract_keywords(text: str) -> set:
            # Tokenize and filter
            words = re.findall(r'\b[a-z]+\b', text.lower())
            # Keep words that are not stop words and have length > 2
            return {w for w in words if w not in stop_words and len(w) > 2}
        
        keywords1 = extract_keywords(query1)
        keywords2 = extract_keywords(query2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = keywords1 & keywords2
        union = keywords1 | keywords2
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Also check if key medical terms from original appear in new query
        # This catches cases like "diabetes" appearing in both
        overlap_ratio = len(intersection) / len(keywords1) if keywords1 else 0.0
        
        return max(jaccard, overlap_ratio)
    
    def _make_decision(
        self,
        has_context_reference: bool,
        reference_type: Optional[str],
        keyword_overlap: float,
        embedding_similarity: float,
        has_new_topic_pattern: bool,
        current_query: str,
        session_context: SessionContext,
    ) -> RouterDecision:
        """
        Make routing decision based on multiple signals.
        """
        # Rule 1: Explicit context references → Route C (context_qa)
        if has_context_reference and reference_type in ["deictic", "explicit_reference", "clarification", "comparison"]:
            logger.info(f"ROUTER_DECISION route=context_qa reason=context_reference ({reference_type})")
            return RouterDecision(
                route="context_qa",
                reasoning=f"Query contains {reference_type} reference to previous results",
                similarity_score=embedding_similarity,
            )
        
        # Rule 2: High keyword overlap + context reference → Route C
        if keyword_overlap >= 0.3 and has_context_reference:
            logger.info(f"ROUTER_DECISION route=context_qa reason=keyword_overlap+reference")
            return RouterDecision(
                route="context_qa",
                reasoning=f"High keyword overlap ({keyword_overlap:.2f}) with context reference",
                similarity_score=embedding_similarity,
            )
        
        # Rule 3: New topic pattern + low keyword overlap → Route A (full_graph)
        if has_new_topic_pattern and keyword_overlap < 0.2:
            logger.info(f"ROUTER_DECISION route=full_graph reason=new_topic_pattern")
            return RouterDecision(
                route="full_graph",
                reasoning="Query appears to be a new topic",
                similarity_score=embedding_similarity,
            )
        
        # Rule 4: Moderate keyword overlap, no explicit reference → Route B (augmented)
        if 0.2 <= keyword_overlap < 0.5 and not has_context_reference:
            # Extract the new focus from the query
            follow_up_focus = self._extract_follow_up_focus(current_query, session_context.original_query)
            logger.info(f"ROUTER_DECISION route=augmented_context reason=related_topic focus={follow_up_focus}")
            return RouterDecision(
                route="augmented_context",
                reasoning=f"Related topic with new focus: {follow_up_focus}",
                similarity_score=embedding_similarity,
                follow_up_focus=follow_up_focus,
            )
        
        # Rule 5: High keyword overlap without explicit reference → Route C
        if keyword_overlap >= 0.5:
            logger.info(f"ROUTER_DECISION route=context_qa reason=high_keyword_overlap ({keyword_overlap:.2f})")
            return RouterDecision(
                route="context_qa",
                reasoning=f"High keyword overlap ({keyword_overlap:.2f}) suggests related follow-up",
                similarity_score=embedding_similarity,
            )
        
        # Rule 6: Low keyword overlap, no references → Route A (new topic)
        if keyword_overlap < 0.2 and not has_context_reference:
            logger.info(f"ROUTER_DECISION route=full_graph reason=low_overlap_no_reference")
            return RouterDecision(
                route="full_graph",
                reasoning="Low keyword overlap, appears to be new topic",
                similarity_score=embedding_similarity,
            )
        
        # Ambiguous case: Use LLM for classification
        logger.info("ROUTER_DECISION using LLM for ambiguous case")
        return self._llm_classify(current_query, session_context, embedding_similarity)
    
    def _extract_follow_up_focus(self, current_query: str, original_query: str) -> Optional[str]:
        """
        Extract the specific new focus/entity from follow-up query.
        E.g., "What about metformin side effects?" → "metformin side effects"
        """
        # Remove common question prefixes
        prefixes_to_remove = [
            r"^what about\s+",
            r"^how about\s+",
            r"^tell me about\s+",
            r"^what are( the)?\s+",
            r"^can you (tell me|explain|find)\s+",
        ]
        
        focus = current_query.lower()
        for prefix in prefixes_to_remove:
            focus = re.sub(prefix, "", focus, flags=re.IGNORECASE)
        
        # Remove trailing question marks and common suffixes
        focus = re.sub(r"\?+$", "", focus).strip()
        focus = re.sub(r"\s+(specifically|in particular|please)$", "", focus, flags=re.IGNORECASE).strip()
        
        # If we got something meaningful, return it
        if len(focus) > 3 and focus != current_query.lower():
            return focus
        
        return None
    
    def _llm_classify(
        self,
        current_query: str,
        session_context: SessionContext,
        precomputed_similarity: float,
    ) -> RouterDecision:
        """
        Use Haiku for nuanced classification in truly ambiguous cases.
        """
        user_message = ROUTER_USER_TEMPLATE.format(
            previous_context=format_previous_context(session_context),
            current_query=current_query,
        )
        
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Parse JSON response
            if not (content.startswith("{") and content.endswith("}")):
                raise ValueError("LLM output is not valid JSON")
            
            data = json.loads(content)
            
            decision = RouterDecision(
                route=data.get("route", "full_graph"),
                reasoning=data.get("reasoning", "LLM classification"),
                similarity_score=data.get("similarity_score", precomputed_similarity),
                follow_up_focus=data.get("follow_up_focus"),
            )
            
            logger.info(
                "ROUTER_LLM_DECISION",
                extra={
                    "route": decision.route,
                    "reasoning": decision.reasoning,
                },
            )
            
            return decision
            
        except Exception as e:
            logger.warning(f"ROUTER_LLM_ERROR: {e}, defaulting to full_graph")
            return RouterDecision(
                route="full_graph",
                reasoning=f"LLM classification failed, defaulting to full search",
                similarity_score=precomputed_similarity,
            )