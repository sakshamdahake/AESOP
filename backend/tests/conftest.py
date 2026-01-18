"""
Shared pytest fixtures for AESOP test suite.

Provides:
- Mock PostgreSQL database
- Mock Redis cache
- Test data factories
- Common test utilities
"""

import pytest
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import Generator, Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock
import fakeredis

# Add backend to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Test configuration values."""
    return {
        "database_url": "postgresql://test:test@localhost:5432/test_db",
        "redis_url": "redis://localhost:6379/15",
        "max_conversations_per_user": 100,
        "max_messages_per_conversation": 500,
        "cache_ttl_seconds": 3600,
        "max_cached_messages": 10,
    }


# ============================================================================
# ID Generation Fixtures
# ============================================================================

@pytest.fixture
def conversation_id() -> str:
    """Generate a valid conversation UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def anonymous_id() -> str:
    """Generate a valid anonymous user ID."""
    return f"anon-{uuid.uuid4().hex[:16]}"


@pytest.fixture
def message_id() -> str:
    """Generate a valid message UUID."""
    return str(uuid.uuid4())


@pytest.fixture
def research_context_id() -> str:
    """Generate a valid research context UUID."""
    return str(uuid.uuid4())


# ============================================================================
# Mock Redis Fixtures
# ============================================================================

@pytest.fixture
def mock_redis() -> fakeredis.FakeRedis:
    """
    Create a fake Redis instance for testing.
    Behaves like real Redis but runs in-memory.
    """
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_redis_with_data(mock_redis, conversation_id, anonymous_id) -> fakeredis.FakeRedis:
    """
    Redis instance pre-populated with test data.
    """
    from app.schemas.conversation import CachedConversation, MessageRecord
    
    # Create cached conversation
    cached = CachedConversation(
        conversation_id=conversation_id,
        anonymous_id=anonymous_id,
        title="Test Conversation",
        title_generated=True,
        recent_messages=[
            MessageRecord(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="user",
                content="Hello!",
                sequence_num=1,
                created_at=datetime.now(timezone.utc),
            ),
            MessageRecord(
                id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                role="assistant",
                content="Hi! I'm AESOP.",
                response_type="chat",
                sequence_num=2,
                created_at=datetime.now(timezone.utc),
            ),
        ],
        message_count=2,
        created_at=datetime.now(timezone.utc),
    )
    
    # Store in Redis
    key = f"aesop:conv:{conversation_id}"
    mock_redis.setex(key, 3600, cached.to_redis())
    
    return mock_redis


# ============================================================================
# Mock Database Fixtures
# ============================================================================

@pytest.fixture
def mock_db_connection():
    """
    Mock PostgreSQL connection that tracks queries.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Track executed queries
    mock_cursor.executed_queries = []
    
    def track_execute(query, params=None):
        mock_cursor.executed_queries.append({
            "query": query,
            "params": params,
        })
    
    mock_cursor.execute = MagicMock(side_effect=track_execute)
    mock_cursor.fetchone = MagicMock(return_value=None)
    mock_cursor.fetchall = MagicMock(return_value=[])
    mock_cursor.rowcount = 0
    
    mock_conn.cursor = MagicMock(return_value=mock_cursor)
    mock_conn.commit = MagicMock()
    mock_conn.rollback = MagicMock()
    
    return mock_conn, mock_cursor


@pytest.fixture
def mock_db_pool(mock_db_connection):
    """
    Mock connection pool that returns mock connections.
    """
    mock_conn, mock_cursor = mock_db_connection
    
    mock_pool = MagicMock()
    mock_pool.getconn = MagicMock(return_value=mock_conn)
    mock_pool.putconn = MagicMock()
    mock_pool.closeall = MagicMock()
    
    return mock_pool


# ============================================================================
# Test Data Factories
# ============================================================================

@pytest.fixture
def sample_conversation(conversation_id, anonymous_id) -> Dict[str, Any]:
    """Create a sample conversation record."""
    now = datetime.now(timezone.utc)
    return {
        "id": conversation_id,
        "anonymous_id": anonymous_id,
        "title": "Test Conversation",
        "title_generated": False,
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }


@pytest.fixture
def sample_messages(conversation_id) -> List[Dict[str, Any]]:
    """Create sample message records."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": "user",
            "content": "What are treatments for diabetes?",
            "response_type": None,
            "sequence_num": 1,
            "created_at": now,
        },
        {
            "id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": "There are several treatments for diabetes...",
            "response_type": "research",
            "sequence_num": 2,
            "created_at": now + timedelta(seconds=5),
        },
    ]


@pytest.fixture
def sample_research_context(conversation_id, research_context_id) -> Dict[str, Any]:
    """Create a sample research context record."""
    return {
        "id": research_context_id,
        "conversation_id": conversation_id,
        "query": "What are treatments for diabetes?",
        "query_embedding": [0.1] * 1536,  # Titan embedding dimension
        "synthesis_summary": "Based on the research...",
        "route_taken": "full_graph",
        "intent": "research",
        "papers_count": 5,
        "critic_decision": "sufficient",
        "avg_quality": 0.75,
        "discard_ratio": 0.2,
        "created_at": datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_papers(research_context_id) -> List[Dict[str, Any]]:
    """Create sample paper records."""
    return [
        {
            "id": str(uuid.uuid4()),
            "research_context_id": research_context_id,
            "pmid": "12345678",
            "title": "Metformin efficacy in Type 2 Diabetes",
            "abstract": "This study evaluated...",
            "journal": "Diabetes Care",
            "publication_year": 2023,
            "relevance_score": 0.85,
            "methodology_score": 0.80,
            "quality_score": 0.825,
            "recommendation": "keep",
        },
        {
            "id": str(uuid.uuid4()),
            "research_context_id": research_context_id,
            "pmid": "87654321",
            "title": "Insulin therapy outcomes",
            "abstract": "A randomized controlled trial...",
            "journal": "NEJM",
            "publication_year": 2022,
            "relevance_score": 0.90,
            "methodology_score": 0.85,
            "quality_score": 0.875,
            "recommendation": "keep",
        },
    ]


# ============================================================================
# Edge Case Data Fixtures
# ============================================================================

@pytest.fixture
def edge_case_messages() -> List[str]:
    """Messages that might break the system."""
    return [
        "",  # Empty
        " ",  # Whitespace only
        "\n\t\r",  # Only control characters
        "a" * 5000,  # Very long message (over limit)
        "Hello! 👋🏽 🧬 💊",  # Emojis
        "<script>alert('xss')</script>",  # XSS attempt
        "'; DROP TABLE conversations; --",  # SQL injection attempt
        "{{template}}",  # Template injection
        "\x00\x01\x02",  # Null bytes
        "Hello\nWorld\tTab",  # Newlines and tabs
        "مرحبا",  # Arabic
        "你好",  # Chinese
        "🔬" * 100,  # Many emojis
        None,  # Null value
    ]


@pytest.fixture
def edge_case_ids() -> List[str]:
    """IDs that might break the system."""
    return [
        "",  # Empty
        " ",  # Whitespace
        "not-a-uuid",  # Invalid UUID format
        "00000000-0000-0000-0000-000000000000",  # Nil UUID
        "../../../etc/passwd",  # Path traversal
        "a" * 500,  # Very long ID
        "<script>",  # XSS in ID
        "'; DROP TABLE",  # SQL injection in ID
        None,  # Null
        123,  # Wrong type (int)
        ["array"],  # Wrong type (list)
    ]


@pytest.fixture
def boundary_values() -> Dict[str, Any]:
    """Boundary values for testing limits."""
    return {
        "min_message_length": 1,
        "max_message_length": 2000,
        "min_title_length": 1,
        "max_title_length": 255,
        "max_conversations": 100,
        "max_messages": 500,
        "max_limit": 100,
        "min_limit": 1,
        "max_offset": 10000,
    }


# ============================================================================
# Mock LLM Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """
    Mock LLM client that returns predictable responses.
    """
    mock = MagicMock()
    
    def mock_invoke(messages):
        # Determine response based on input
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "").lower()
                break
        
        response = MagicMock()
        
        # Intent classification
        if "classify" in user_content or "intent" in user_content:
            response.content = '{"intent": "chat", "confidence": 0.95, "reasoning": "Test response"}'
        # Title generation
        elif "title" in user_content or "generate a title" in user_content:
            response.content = "Test conversation title"
        # Chat response
        else:
            response.content = "This is a test response from the mock LLM."
        
        return response
    
    mock.invoke = MagicMock(side_effect=mock_invoke)
    return mock


@pytest.fixture
def mock_llm_timeout():
    """Mock LLM that always times out."""
    mock = MagicMock()
    
    def timeout_invoke(messages):
        import time
        time.sleep(35)  # Exceed timeout
        raise TimeoutError("LLM request timed out")
    
    mock.invoke = MagicMock(side_effect=timeout_invoke)
    return mock


@pytest.fixture
def mock_llm_error():
    """Mock LLM that always raises an error."""
    mock = MagicMock()
    mock.invoke = MagicMock(side_effect=Exception("LLM service unavailable"))
    return mock


@pytest.fixture
def mock_llm_invalid_json():
    """Mock LLM that returns invalid JSON."""
    mock = MagicMock()
    response = MagicMock()
    response.content = "This is not valid JSON {{{{"
    mock.invoke = MagicMock(return_value=response)
    return mock


# ============================================================================
# HTTP Client Fixtures (for API tests)
# ============================================================================

@pytest.fixture
def test_client():
    """
    Create a test client for API testing.
    """
    from fastapi.testclient import TestClient
    from app.main import app
    
    return TestClient(app)


@pytest.fixture
def authenticated_headers(anonymous_id) -> Dict[str, str]:
    """Headers with anonymous ID for authenticated requests."""
    return {
        "X-Anonymous-Id": anonymous_id,
        "Content-Type": "application/json",
    }


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_singletons():
    """
    Reset singleton instances between tests.
    """
    yield
    
    # Reset database service singleton
    from app.services import database
    database._db_service = None
    
    # Reset chat memory service singleton
    from app.services import chat_memory
    chat_memory._chat_memory_service = None
    
    # Reset title generator singleton
    try:
        from app.agents.title import agent
        agent._title_generator = None
    except ImportError:
        pass


# ============================================================================
# Time Manipulation Fixtures
# ============================================================================

@pytest.fixture
def frozen_time():
    """
    Fixture for freezing time during tests.
    Usage: with frozen_time("2024-01-15 12:00:00"): ...
    """
    from freezegun import freeze_time
    return freeze_time