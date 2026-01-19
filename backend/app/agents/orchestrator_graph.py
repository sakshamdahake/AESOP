"""
Orchestrator Graph - Session-aware LangGraph with Intent classification.

Flow:
  Intent → (Chat | Utility | Router)
  Router → (Route A | Route B | Route C)
  
Routes:
- Chat: General conversation (no research)
- Utility: Reformat existing output
- Route A: Full Graph (Scout → Critic → Synthesizer)
- Route B: Augmented Context (Scout → Merge → Synthesizer)  
- Route C: Context Q&A (direct answer from cache)
"""

from langgraph.graph import StateGraph, END
from datetime import datetime

from app.agents.state import OrchestratorState, Paper
from app.agents.intent.node import intent_node
from app.agents.chat.node import chat_node
from app.agents.utility.node import utility_node
from app.agents.router.node import router_node
from app.agents.scout.agent import scout_node as original_scout_node
from app.agents.critic.node import critic_node as original_critic_node
from app.agents.synthesizer.agent import synthesizer_node as original_synthesizer_node
from app.agents.context_qa.node import context_qa_node

from app.schemas.session import SessionContext, CachedPaper
from app.services.session import get_session_service, SESSION_TTL_SECONDS
from app.embeddings.bedrock import embed_query
from app.logging import logger


# ============================
# Adapter Nodes (unchanged from before)
# ============================

def scout_node(state: OrchestratorState) -> OrchestratorState:
    """
    Adapter: Run original scout_node with OrchestratorState.
    For Route B, modify query to focus on follow_up_focus.
    """
    from app.agents.state import AgentState
    
    # Determine query to use
    query = state.query
    if (
        state.router_decision 
        and state.router_decision.route == "augmented_context"
        and state.router_decision.follow_up_focus
    ):
        # Route B: Combine original context with specific focus
        original_query = state.session_context.original_query if state.session_context else ""
        query = f"{original_query} {state.router_decision.follow_up_focus}"
        logger.info(f"SCOUT_AUGMENTED_QUERY query='{query[:60]}...'")
    
    # Create minimal AgentState for scout
    agent_state = AgentState(
        query=query,
        iteration_count=state.iteration_count,
        max_iterations=state.max_iterations,
    )
    
    # Run original scout
    result = original_scout_node(agent_state)
    
    # Transfer results to OrchestratorState
    new_state = state.model_copy(deep=True)
    new_state.expanded_queries = result.expanded_queries
    new_state.retrieved_papers = result.retrieved_papers
    new_state.iteration_count = result.iteration_count
    
    return new_state


def critic_node(state: OrchestratorState) -> OrchestratorState:
    """
    Adapter: Run original critic_node with OrchestratorState.
    """
    from app.agents.state import AgentState
    
    # Create AgentState with retrieved papers
    agent_state = AgentState(
        query=state.query,
        retrieved_papers=state.retrieved_papers,
        iteration_count=state.iteration_count,
        max_iterations=state.max_iterations,
    )
    
    # Run original critic
    result = original_critic_node(agent_state)
    
    # Transfer results
    new_state = state.model_copy(deep=True)
    new_state.grades = result.grades
    new_state.critic_decision = result.critic_decision
    new_state.critic_explanation = result.critic_explanation
    new_state.avg_quality = result.avg_quality
    new_state.discard_ratio = result.discard_ratio
    
    return new_state


def synthesizer_node(state: OrchestratorState) -> OrchestratorState:
    """
    Adapter: Run original synthesizer_node with OrchestratorState.
    Uses merged_papers for Route B, retrieved_papers for Route A.
    """
    from app.agents.state import AgentState
    
    # Determine which papers to use
    if state.merged_papers:
        # Route B: Use merged papers
        papers = [
            Paper(
                pmid=p.pmid,
                title=p.title,
                abstract=p.abstract,
                publication_year=p.publication_year,
                journal=p.journal,
            )
            for p in state.merged_papers
        ]
        grades = state.grades  # May be empty for Route B
    else:
        # Route A: Use retrieved papers
        papers = state.retrieved_papers
        grades = state.grades
    
    agent_state = AgentState(
        query=state.query,
        retrieved_papers=papers,
        grades=grades,
    )
    
    # Run original synthesizer
    result = original_synthesizer_node(agent_state)
    
    new_state = state.model_copy(deep=True)
    new_state.synthesis_output = result.synthesis_output
    
    return new_state


def merge_node(state: OrchestratorState) -> OrchestratorState:
    """
    Route B: Merge cached papers with newly retrieved papers.
    Deduplicates by PMID, sorts by quality.
    """
    cached_papers = []
    if state.session_context and state.session_context.retrieved_papers:
        cached_papers = state.session_context.retrieved_papers
    
    # Convert new papers to CachedPaper format
    new_papers = [
        CachedPaper(
            pmid=p.pmid,
            title=p.title,
            abstract=p.abstract,
            publication_year=p.publication_year,
            journal=p.journal,
            quality_score=0.5,  # Default for ungraded
        )
        for p in state.retrieved_papers
    ]
    
    # Deduplicate by PMID
    seen_pmids = set()
    merged = []
    
    # Prioritize cached papers (already graded)
    for paper in cached_papers:
        if paper.pmid not in seen_pmids:
            seen_pmids.add(paper.pmid)
            merged.append(paper)
    
    # Add new papers
    for paper in new_papers:
        if paper.pmid not in seen_pmids:
            seen_pmids.add(paper.pmid)
            merged.append(paper)
    
    # Sort by quality (cached papers with scores first)
    merged.sort(key=lambda p: p.quality_score or 0, reverse=True)
    
    logger.info(
        "MERGE_NODE",
        extra={
            "cached_count": len(cached_papers),
            "new_count": len(new_papers),
            "merged_count": len(merged),
        },
    )
    
    new_state = state.model_copy(deep=True)
    new_state.merged_papers = merged[:15]  # Cap at 15 papers
    
    return new_state


def save_session_node(state: OrchestratorState) -> OrchestratorState:
    """
    Save session context for future queries.
    Chat and Utility routes don't update session (just extend TTL if exists).
    """
    session_service = get_session_service()
    
    route_taken = state.route_taken
    session_id = state.session_id
    query = state.query
    
    # Chat/Utility/Context QA: Just extend TTL if session exists
    if route_taken in ("chat", "utility", "context_qa"):
        if session_id:
            session_service.extend_ttl(session_id)
            logger.info(f"SESSION_TTL_EXTENDED session_id={session_id} route={route_taken}")
        return state
    
    # Route A/B: Save full context
    if not session_id:
        # No session to save to
        return state
    
    query_embedding = embed_query(query)
    
    # Convert papers to CachedPaper format with grades
    cached_papers = []
    grade_map = {g.pmid: g for g in state.grades} if state.grades else {}
    
    papers_to_cache = state.merged_papers if state.merged_papers else []
    if not papers_to_cache:
        papers_to_cache = [
            CachedPaper(
                pmid=p.pmid,
                title=p.title,
                abstract=p.abstract,
                publication_year=p.publication_year,
                journal=p.journal,
            )
            for p in state.retrieved_papers
        ]
    
    for paper in papers_to_cache:
        grade = grade_map.get(paper.pmid)
        if isinstance(paper, CachedPaper):
            cached = paper
        else:
            cached = CachedPaper(
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                publication_year=paper.publication_year,
                journal=paper.journal,
            )
        
        if grade:
            cached.relevance_score = grade.relevance_score
            cached.methodology_score = grade.methodology_score
            cached.quality_score = (grade.relevance_score + grade.methodology_score) / 2
            cached.recommendation = grade.recommendation.value if hasattr(grade.recommendation, 'value') else str(grade.recommendation)
        
        cached_papers.append(cached)
    
    # Get or create session context
    existing = state.session_context
    turn_count = (existing.turn_count + 1) if existing else 1
    created_at = existing.created_at if existing else datetime.utcnow()

    # Truncate synthesis for caching
    synthesis_summary = (state.synthesis_output or "")[:1500]

    context = SessionContext(
        session_id=session_id,
        original_query=query,
        query_embedding=query_embedding,
        retrieved_papers=cached_papers[:15],  # Cap storage
        synthesis_summary=synthesis_summary,
        turn_count=turn_count,
        created_at=created_at,
    )

    # Only cache in Redis (don't save to DB - main.py will do it with messages)
    try:
        session_service._redis.setex(
            session_service._redis_key(session_id),
            SESSION_TTL_SECONDS,
            context.to_redis(),
        )
        logger.debug(
            "SESSION_CONTEXT_CACHED_REDIS",
            extra={"session_id": session_id}
        )
    except Exception as e:
        logger.warning(
            "SESSION_CONTEXT_CACHE_ERROR",
            extra={"session_id": session_id, "error": str(e)}
        )

    return state



# ============================
# Intent-Based Routing
# ============================

def route_by_intent(state: OrchestratorState) -> str:
    """
    Route based on classified intent.
    """
    intent = state.intent
    
    if intent == "chat":
        return "chat"
    elif intent == "utility":
        return "utility"
    elif intent in ("research", "followup_research"):
        return "router"
    else:
        # Default to router (research)
        return "router"


def route_by_router_decision(state: OrchestratorState) -> str:
    """
    Route based on Router agent's decision (for research intents).
    """
    if state.router_decision is None:
        return "scout"  # Fallback
    
    route = state.router_decision.route
    
    if route == "full_graph":
        return "scout"
    elif route == "augmented_context":
        return "scout_augmented"
    elif route == "context_qa":
        return "context_qa"
    else:
        return "scout"  # Fallback


def crag_routing(state: OrchestratorState) -> str:
    """CRAG loop routing after critic (Route A only)."""
    if (
        state.critic_decision == "retrieve_more"
        and state.iteration_count < state.max_iterations
    ):
        return "scout"
    return "synthesizer"


# ============================
# Graph Construction
# ============================

def build_orchestrator_graph():
    """
    Build the intent-aware orchestrator graph.
    
    Flow:
        Intent → Chat (if chat intent)
               → Utility (if utility intent)
               → Router (if research/followup intent)
                   → Scout → Critic → Synthesizer (Route A)
                   → Scout → Merge → Synthesizer (Route B)
                   → Context Q&A (Route C)
    """
    
    graph = StateGraph(OrchestratorState)
    
    # Add all nodes
    graph.add_node("intent", intent_node)
    graph.add_node("chat", chat_node)
    graph.add_node("utility", utility_node)
    graph.add_node("router", router_node)
    graph.add_node("scout", scout_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("scout_augmented", scout_node)  # Same function, different path
    graph.add_node("merge", merge_node)
    graph.add_node("context_qa", context_qa_node)
    graph.add_node("save_session", save_session_node)
    
    # Entry point: Intent classification
    graph.set_entry_point("intent")
    
    # Intent → (Chat | Utility | Router)
    graph.add_conditional_edges(
        "intent",
        route_by_intent,
        {
            "chat": "chat",
            "utility": "utility",
            "router": "router",
        },
    )
    
    # Chat → Save Session → END
    graph.add_edge("chat", "save_session")
    
    # Utility → Save Session → END
    graph.add_edge("utility", "save_session")
    
    # Router → Routes
    graph.add_conditional_edges(
        "router",
        route_by_router_decision,
        {
            "scout": "scout",
            "scout_augmented": "scout_augmented",
            "context_qa": "context_qa",
        },
    )
    
    # Route A: Full Graph with CRAG loop
    graph.add_edge("scout", "critic")
    graph.add_conditional_edges(
        "critic",
        crag_routing,
        {
            "scout": "scout",
            "synthesizer": "synthesizer",
        },
    )
    
    # Route B: Augmented Context (no critic loop)
    graph.add_edge("scout_augmented", "merge")
    graph.add_edge("merge", "synthesizer")
    
    # Route C: Context Q&A
    graph.add_edge("context_qa", "save_session")
    
    # Synthesizer → Save Session → END
    graph.add_edge("synthesizer", "save_session")
    graph.add_edge("save_session", END)
    
    return graph.compile()


# Compiled graph
orchestrator_graph = build_orchestrator_graph()