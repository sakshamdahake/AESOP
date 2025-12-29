"""
Hybrid Intent Classifier Agent.

Strategy:
1. Fast-path: Trivial patterns (greetings, thanks) → instant classification
2. Medical keyword detection → likely research
3. LLM classification → smart, context-aware decision for everything else

This approach minimizes LLM calls while maintaining high accuracy.
"""

import json
import re
import time
from typing import Optional, Literal, Tuple

from app.schemas.session import SessionContext
from app.agents.intent.prompts import (
    INTENT_SYSTEM_PROMPT,
    INTENT_USER_TEMPLATE,
    format_session_for_intent,
)
from app.logging import logger


IntentType = Literal["research", "followup_research", "chat", "utility"]


class IntentClassifier:
    """
    Hybrid intent classifier combining:
    1. Regex fast-path for trivial cases (no LLM cost)
    2. Keyword hints for confidence adjustment
    3. LLM classification for nuanced decisions
    """
    
    # =========================================================================
    # FAST-PATH PATTERNS (skip LLM entirely for these)
    # =========================================================================
    
    # Trivial chat - extremely common, no ambiguity
    TRIVIAL_CHAT = [
        r"^hi+[.!]?$",
        r"^hello+[.!]?$",
        r"^hey+[.!]?$",
        r"^yo[.!]?$",
        r"^thanks?(\s+you)?[.!]?$",
        r"^thank\s+you[.!]?$",
        r"^thx[.!]?$",
        r"^ty[.!]?$",
        r"^bye[.!]?$",
        r"^goodbye[.!]?$",
        r"^ok(ay)?[.!]?$",
        r"^yes[.!]?$",
        r"^no[.!]?$",
        r"^yeah[.!]?$",
        r"^nope[.!]?$",
        r"^cool[.!]?$",
        r"^great[.!]?$",
        r"^nice[.!]?$",
        r"^awesome[.!]?$",
        r"^perfect[.!]?$",
        r"^got\s*it[.!]?$",
        r"^i\s+see[.!]?$",
        r"^understood[.!]?$",
        r"^sure[.!]?$",
        r"^lol[.!]?$",
        r"^haha[.!]?$",
        r"^wow[.!]?$",
        r"^oh[.!]?$",
        r"^hmm+[.!]?$",
    ]
    
    # =========================================================================
    # KEYWORD HINTS (used to guide LLM or quick-check)
    # =========================================================================
    
    # Strong medical indicators - if present, likely research
    MEDICAL_KEYWORDS = {
        # Conditions
        "diabetes", "cancer", "tumor", "asthma", "alzheimer", "parkinson",
        "arthritis", "hypertension", "stroke", "heart disease", "covid",
        "coronavirus", "influenza", "pneumonia", "hepatitis", "hiv", "aids",
        "depression", "anxiety", "schizophrenia", "bipolar", "adhd", "autism",
        "epilepsy", "migraine", "obesity", "anemia", "leukemia", "lymphoma",
        "melanoma", "cirrhosis", "fibrosis", "thrombosis", "embolism",
        
        # Treatments/Drugs
        "treatment", "therapy", "medication", "drug", "medicine", "vaccine",
        "antibiotic", "chemotherapy", "radiation", "surgery", "transplant",
        "metformin", "insulin", "ibuprofen", "aspirin", "statin", "steroid",
        "antidepressant", "antipsychotic", "painkiller", "opioid",
        
        # Medical terms
        "symptom", "diagnosis", "prognosis", "etiology", "pathology",
        "clinical", "patient", "disease", "disorder", "syndrome", "condition",
        "chronic", "acute", "benign", "malignant", "remission", "relapse",
        "dosage", "side effect", "adverse effect", "contraindication",
        
        # Research terms
        "study", "trial", "rct", "randomized", "placebo", "efficacy",
        "mortality", "morbidity", "incidence", "prevalence", "meta-analysis",
        "systematic review", "pubmed", "clinical trial",
        
        # Body parts/systems
        "blood", "liver", "kidney", "lung", "brain", "heart", "bone",
        "muscle", "nerve", "artery", "vein", "immune", "hormone",
    }
    
    # System/meta indicators - if present, likely chat
    SYSTEM_KEYWORDS = {
        "who are you", "what are you", "your name", "about yourself",
        "what can you do", "how do you work", "how does this work",
        "are you a bot", "are you ai", "are you real", "are you human",
        "can i chat", "can we chat", "can i talk", "can we talk",
        "how long can", "is this free", "do you remember", "your purpose",
        "help me understand", "what is aesop", "what is this",
    }
    
    # Followup indicators - references to previous content
    FOLLOWUP_KEYWORDS = {
        "these studies", "those studies", "the studies", "the papers",
        "these papers", "those papers", "these results", "those results",
        "the findings", "these findings", "first paper", "second paper",
        "first study", "second study", "compare them", "compare these",
        "which one", "which study", "tell me more", "more details",
        "elaborate", "explain more", "go deeper",
    }
    
    # Utility indicators - reformatting requests
    UTILITY_KEYWORDS = {
        "make it shorter", "make it simpler", "make it longer",
        "bullet points", "numbered list", "summarize it", "simplify it",
        "convert to", "reformat", "just the conclusion", "just the summary",
        "key points only", "shorter version", "simpler version",
    }
    
    def __init__(self, llm_client):
        """Initialize with LLM client for smart classification."""
        self.llm = llm_client
        self._trivial_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.TRIVIAL_CHAT
        ]
    
    def classify(
        self,
        message: str,
        session_context: Optional[SessionContext] = None,
    ) -> Tuple[IntentType, float, str]:
        """
        Classify user intent using hybrid approach.
        
        Returns:
            Tuple of (intent, confidence, reasoning)
        """
        message = message.strip()
        message_lower = message.lower()
        
        start_time = time.time()
        
        # ---------------------------------------------------------------------
        # STAGE 1: Fast-path for trivial messages (no LLM needed)
        # ---------------------------------------------------------------------
        
        # Empty/very short
        if len(message) < 2:
            return "chat", 1.0, "Empty message"
        
        # Trivial chat patterns
        if self._is_trivial_chat(message_lower):
            self._log_classification("fast_path", "chat", 0.99, message, start_time)
            return "chat", 0.99, "Trivial chat (greeting/thanks/acknowledgment)"
        
        # ---------------------------------------------------------------------
        # STAGE 2: Keyword pre-analysis (helps LLM and catches obvious cases)
        # ---------------------------------------------------------------------
        
        has_medical = self._has_keywords(message_lower, self.MEDICAL_KEYWORDS)
        has_system = self._has_keywords(message_lower, self.SYSTEM_KEYWORDS)
        has_followup = self._has_keywords(message_lower, self.FOLLOWUP_KEYWORDS)
        has_utility = self._has_keywords(message_lower, self.UTILITY_KEYWORDS)
        has_session = session_context is not None
        has_output = has_session and bool(session_context.synthesis_summary)
        
        # Log keyword analysis
        logger.debug(
            "INTENT_KEYWORD_ANALYSIS",
            extra={
                "has_medical": has_medical,
                "has_system": has_system,
                "has_followup": has_followup,
                "has_utility": has_utility,
                "has_session": has_session,
            },
        )
        
        # ---------------------------------------------------------------------
        # STAGE 3: Quick decisions for clear-cut cases
        # ---------------------------------------------------------------------
        
        # Clear system/meta question (no medical content)
        if has_system and not has_medical:
            self._log_classification("keyword", "chat", 0.95, message, start_time)
            return "chat", 0.95, "System/meta question detected"
        
        # Clear utility request with output
        if has_utility and has_output:
            self._log_classification("keyword", "utility", 0.92, message, start_time)
            return "utility", 0.92, "Utility request with existing output"
        
        # Clear followup with session
        if has_followup and has_session:
            self._log_classification("keyword", "followup_research", 0.92, message, start_time)
            return "followup_research", 0.92, "Follow-up reference with session context"
        
        # Very short message without medical keywords -> chat
        if len(message.split()) <= 4 and not has_medical:
            self._log_classification("heuristic", "chat", 0.85, message, start_time)
            return "chat", 0.85, "Short message without medical content"
        
        # ---------------------------------------------------------------------
        # STAGE 4: LLM classification for ambiguous cases
        # ---------------------------------------------------------------------
        
        intent, confidence, reasoning = self._llm_classify(message, session_context)
        
        # Validate context requirements
        intent = self._validate_context(intent, session_context)
        
        self._log_classification("llm", intent, confidence, message, start_time)
        
        return intent, confidence, reasoning
    
    def _is_trivial_chat(self, message_lower: str) -> bool:
        """Check if message matches trivial chat patterns."""
        # Remove punctuation for matching
        cleaned = re.sub(r'[^\w\s]', '', message_lower).strip()
        
        for pattern in self._trivial_patterns:
            if pattern.match(cleaned) or pattern.match(message_lower):
                return True
        
        return False
    
    def _has_keywords(self, message_lower: str, keywords: set) -> bool:
        """Check if message contains any keywords from the set."""
        return any(kw in message_lower for kw in keywords)
    
    def _llm_classify(
        self,
        message: str,
        session_context: Optional[SessionContext],
    ) -> Tuple[IntentType, float, str]:
        """Use LLM for nuanced classification."""
        session_info = format_session_for_intent(session_context)
        
        user_message = INTENT_USER_TEMPLATE.format(
            has_session=session_info["has_session"],
            previous_query=session_info["previous_query"],
            turn_count=session_info["turn_count"],
            current_message=message,
        )
        
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Parse JSON response
            data = self._parse_llm_response(content)
            
            intent = data.get("intent", "chat")
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.7))))
            reasoning = data.get("reasoning", "LLM classification")
            
            # Validate intent value
            if intent not in ("research", "followup_research", "chat", "utility"):
                logger.warning(f"INTENT_INVALID_VALUE: {intent}, defaulting to chat")
                intent = "chat"
            
            return intent, confidence, reasoning
            
        except Exception as e:
            logger.warning(
                "INTENT_LLM_ERROR",
                extra={"error": str(e), "user_input": message[:50]},
            )
            # Safe default
            return "chat", 0.5, f"LLM error, defaulting to chat"
    
    def _parse_llm_response(self, content: str) -> dict:
        """Robustly parse JSON from LLM response."""
        # Clean up common issues
        content = content.strip()
        
        # Remove markdown code fences
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\s*', '', content)
            content = re.sub(r'\s*```$', '', content)
            content = content.strip()
        
        # Direct JSON parse
        if content.startswith("{") and content.endswith("}"):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        
        # Extract JSON from text
        json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Last resort: try to find key-value pairs
        intent_match = re.search(r'"intent"\s*:\s*"(\w+)"', content)
        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content)
        
        if intent_match:
            return {
                "intent": intent_match.group(1),
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.7,
                "reasoning": reasoning_match.group(1) if reasoning_match else "Parsed from text",
            }
        
        raise ValueError(f"Could not parse LLM response: {content[:200]}")
    
    def _validate_context(
        self,
        intent: str,
        session_context: Optional[SessionContext],
    ) -> str:
        """Validate intent makes sense given context."""
        # followup_research requires session
        if intent == "followup_research" and not session_context:
            logger.debug("INTENT_VALIDATE: followup->research (no session)")
            return "research"
        
        # utility requires previous output
        if intent == "utility":
            if not session_context or not session_context.synthesis_summary:
                logger.debug("INTENT_VALIDATE: utility->chat (no output)")
                return "chat"
        
        return intent
    
    def _log_classification(
        self,
        method: str,
        intent: str,
        confidence: float,
        message: str,
        start_time: float,
    ):
        """Log classification result."""
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "INTENT_CLASSIFIED",
            extra={
                "method": method,
                "intent": intent,
                "confidence": round(confidence, 2),
                "elapsed_ms": round(elapsed_ms, 1),
                "user_input": message[:50],
            },
        )