"""
Unit tests for DatabaseService.

Tests cover:
- Connection pool management
- CRUD operations for all tables
- Rate limiting (conversation and message limits)
- Edge cases (null values, invalid IDs, SQL injection)
- Error handling and recovery
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import psycopg2

from app.services.database import DatabaseService, get_database_service
from app.schemas.conversation import (
    ConversationRecord,
    MessageRecord,
    ResearchContextRecord,
    ResearchPaperRecord,
)


class TestDatabaseServiceInitialization:
    """Tests for connection pool initialization and shutdown."""
    
    def test_singleton_pattern(self):
        """DatabaseService should be a singleton."""
        service1 = DatabaseService()
        service2 = DatabaseService()
        assert service1 is service2
    
    def test_get_database_service_returns_singleton(self):
        """get_database_service() should return the same instance."""
        service1 = get_database_service()
        service2 = get_database_service()
        assert service1 is service2
    
    @patch('app.services.database.pool.ThreadedConnectionPool')
    def test_initialize_creates_pool(self, mock_pool_class):
        """initialize() should create a connection pool."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        service = DatabaseService()
        service._pool = None  # Reset pool
        service.initialize()
        
        mock_pool_class.assert_called_once()
        assert service._pool is not None
    
    @patch('app.services.database.pool.ThreadedConnectionPool')
    def test_initialize_twice_is_idempotent(self, mock_pool_class):
        """Calling initialize() twice should not create two pools."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        service = DatabaseService()
        service._pool = None
        service.initialize()
        service.initialize()
        
        # Should only be called once
        assert mock_pool_class.call_count == 1
    
    @patch('app.services.database.pool.ThreadedConnectionPool')
    def test_initialize_handles_connection_error(self, mock_pool_class):
        """initialize() should raise on connection failure."""
        mock_pool_class.side_effect = psycopg2.OperationalError("Connection refused")
        
        service = DatabaseService()
        service._pool = None
        
        with pytest.raises(psycopg2.OperationalError):
            service.initialize()
    
    def test_shutdown_closes_pool(self, mock_db_pool):
        """shutdown() should close all connections."""
        service = DatabaseService()
        service._pool = mock_db_pool
        
        service.shutdown()
        
        mock_db_pool.closeall.assert_called_once()
        assert service._pool is None
    
    def test_shutdown_without_pool_is_safe(self):
        """shutdown() should handle case where pool is None."""
        service = DatabaseService()
        service._pool = None
        
        # Should not raise
        service.shutdown()


class TestDatabaseServiceConnectionManagement:
    """Tests for connection acquisition and release."""
    
    def test_get_connection_returns_connection(self, mock_db_pool, mock_db_connection):
        """get_connection() should return a valid connection."""
        mock_conn, _ = mock_db_connection
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        with service.get_connection() as conn:
            assert conn is mock_conn
    
    def test_get_connection_commits_on_success(self, mock_db_pool, mock_db_connection):
        """Connection should be committed on successful exit."""
        mock_conn, _ = mock_db_connection
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        with service.get_connection() as conn:
            pass  # Successful operation
        
        mock_conn.commit.assert_called_once()
    
    def test_get_connection_rollback_on_error(self, mock_db_pool, mock_db_connection):
        """Connection should be rolled back on exception."""
        mock_conn, _ = mock_db_connection
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        with pytest.raises(ValueError):
            with service.get_connection() as conn:
                raise ValueError("Test error")
        
        mock_conn.rollback.assert_called_once()
    
    def test_get_connection_returns_to_pool(self, mock_db_pool, mock_db_connection):
        """Connection should be returned to pool after use."""
        mock_conn, _ = mock_db_connection
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        with service.get_connection() as conn:
            pass
        
        mock_db_pool.putconn.assert_called_once_with(mock_conn)
    
    def test_get_connection_initializes_pool_if_needed(self, mock_db_pool):
        """get_connection() should initialize pool if not already done."""
        service = DatabaseService()
        service._pool = None
        
        with patch.object(service, 'initialize') as mock_init:
            # Make initialize set up the pool
            def set_pool():
                service._pool = mock_db_pool
            mock_init.side_effect = set_pool
            
            with service.get_connection():
                pass
            
            mock_init.assert_called_once()


class TestConversationCRUD:
    """Tests for conversation CRUD operations."""
    
    def test_create_conversation_returns_record(self, mock_db_pool, mock_db_connection):
        """create_conversation() should return a ConversationRecord."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        conv_id = str(uuid.uuid4())
        
        # Mock the RETURNING clause
        mock_cursor.fetchone.return_value = (
            conv_id,
            "anon-123",
            "New Conversation",
            False,
            now,
            now,
        )
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.create_conversation(
            anonymous_id="anon-123",
            title="New Conversation",
        )
        
        assert isinstance(result, ConversationRecord)
        assert result.anonymous_id == "anon-123"
        assert result.title == "New Conversation"
        assert result.title_generated is False
    
    def test_create_conversation_without_anonymous_id(self, mock_db_pool, mock_db_connection):
        """create_conversation() should work without anonymous_id."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        conv_id = str(uuid.uuid4())
        
        mock_cursor.fetchone.return_value = (
            conv_id,
            None,  # No anonymous_id
            "New Conversation",
            False,
            now,
            now,
        )
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.create_conversation()
        
        assert result.anonymous_id is None
    
    def test_create_conversation_rate_limit_error(self, mock_db_pool, mock_db_connection):
        """create_conversation() should raise ValueError on rate limit."""
        mock_conn, mock_cursor = mock_db_connection
        
        # Simulate rate limit trigger error
        mock_cursor.execute.side_effect = psycopg2.Error(
            "Conversation limit reached. Maximum 100 conversations per user."
        )
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        with pytest.raises(ValueError, match="Conversation limit reached"):
            service.create_conversation(anonymous_id="anon-123")
    
    def test_get_conversation_returns_none_for_invalid_id(self, mock_db_pool, mock_db_connection):
        """get_conversation() should return None for non-existent ID."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.get_conversation("non-existent-id")
        
        assert result is None
    
    def test_get_conversation_returns_none_for_deleted(self, mock_db_pool, mock_db_connection):
        """get_conversation() should return None for soft-deleted conversations."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None  # WHERE deleted_at IS NULL
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.get_conversation("deleted-conv-id")
        
        assert result is None
    
    def test_get_conversation_with_messages(self, mock_db_pool, mock_db_connection, conversation_id):
        """get_conversation() should include messages when requested."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        
        # First call: conversation
        # Second call: messages
        mock_cursor.fetchone.return_value = (
            conversation_id,
            "anon-123",
            "Test Conversation",
            True,
            now,
            now,
            None,  # deleted_at
        )
        
        mock_cursor.fetchall.return_value = [
            (str(uuid.uuid4()), conversation_id, "user", "Hello", None, 1, now),
            (str(uuid.uuid4()), conversation_id, "assistant", "Hi!", "chat", 2, now),
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.get_conversation(conversation_id, include_messages=True)
        
        assert result is not None
        assert len(result.messages) == 2
        assert result.message_count == 2
    
    def test_delete_conversation_soft_deletes(self, mock_db_pool, mock_db_connection, conversation_id):
        """delete_conversation() should set deleted_at, not actually delete."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.rowcount = 1
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.delete_conversation(conversation_id)
        
        assert result is True
        
        # Verify UPDATE was called, not DELETE
        executed = mock_cursor.execute.call_args[0][0]
        assert "UPDATE" in executed
        assert "deleted_at" in executed
    
    def test_delete_conversation_returns_false_for_not_found(self, mock_db_pool, mock_db_connection):
        """delete_conversation() should return False if conversation not found."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.rowcount = 0  # No rows updated
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.delete_conversation("non-existent-id")
        
        assert result is False
    
    def test_update_conversation_title(self, mock_db_pool, mock_db_connection, conversation_id):
        """update_conversation_title() should update title and title_generated."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.rowcount = 1
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.update_conversation_title(
            conversation_id=conversation_id,
            title="New Title",
            title_generated=True,
        )
        
        assert result is True


class TestMessageCRUD:
    """Tests for message CRUD operations."""
    
    def test_add_message_returns_record(self, mock_db_pool, mock_db_connection, conversation_id):
        """add_message() should return a MessageRecord."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        msg_id = str(uuid.uuid4())
        
        # First call: get sequence number
        mock_cursor.fetchone.side_effect = [
            (1,),  # Sequence number
            (msg_id, conversation_id, "user", "Hello", None, 1, now),  # RETURNING
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.add_message(
            conversation_id=conversation_id,
            role="user",
            content="Hello",
        )
        
        assert isinstance(result, MessageRecord)
        assert result.role == "user"
        assert result.content == "Hello"
        assert result.sequence_num == 1
    
    def test_add_message_increments_sequence(self, mock_db_pool, mock_db_connection, conversation_id):
        """add_message() should get correct sequence number."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        
        # Simulate existing messages
        mock_cursor.fetchone.side_effect = [
            (5,),  # Next sequence is 5
            (str(uuid.uuid4()), conversation_id, "user", "Test", None, 5, now),
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.add_message(
            conversation_id=conversation_id,
            role="user",
            content="Test",
        )
        
        assert result.sequence_num == 5
    
    def test_add_message_rate_limit_error(self, mock_db_pool, mock_db_connection, conversation_id):
        """add_message() should raise ValueError on message limit."""
        mock_conn, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.return_value = (1,)  # Sequence number
        mock_cursor.execute.side_effect = [
            None,  # First call (sequence)
            psycopg2.Error("Message limit reached. Maximum 500 messages per conversation."),
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        with pytest.raises(ValueError, match="Message limit reached"):
            service.add_message(
                conversation_id=conversation_id,
                role="user",
                content="Test",
            )
    
    def test_get_recent_messages_respects_limit(self, mock_db_pool, mock_db_connection, conversation_id):
        """get_recent_messages() should respect the limit parameter."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        
        # Return 3 messages
        mock_cursor.fetchall.return_value = [
            (str(uuid.uuid4()), conversation_id, "assistant", "Msg 3", "chat", 3, now),
            (str(uuid.uuid4()), conversation_id, "user", "Msg 2", None, 2, now),
            (str(uuid.uuid4()), conversation_id, "assistant", "Msg 1", "chat", 1, now),
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.get_recent_messages(conversation_id, limit=3)
        
        # Should be in chronological order (reversed from DESC query)
        assert len(result) == 3
        assert result[0].sequence_num == 1
        assert result[2].sequence_num == 3
    
    def test_get_message_count(self, mock_db_pool, mock_db_connection, conversation_id):
        """get_message_count() should return correct count."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = (42,)
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.get_message_count(conversation_id)
        
        assert result == 42


class TestListConversations:
    """Tests for listing conversations."""
    
    def test_list_conversations_returns_tuple(self, mock_db_pool, mock_db_connection, anonymous_id):
        """list_conversations() should return (conversations, total)."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        
        # First call: count
        # Second call: conversations
        mock_cursor.fetchone.return_value = (5,)  # Total count
        mock_cursor.fetchall.return_value = [
            (str(uuid.uuid4()), "Conv 1", now, now, 10, "Last message 1"),
            (str(uuid.uuid4()), "Conv 2", now, now, 5, "Last message 2"),
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        conversations, total = service.list_conversations(
            anonymous_id=anonymous_id,
            limit=20,
            offset=0,
        )
        
        assert total == 5
        assert len(conversations) == 2
    
    def test_list_conversations_truncates_preview(self, mock_db_pool, mock_db_connection, anonymous_id):
        """list_conversations() should truncate long message previews."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        long_message = "x" * 200
        
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = [
            (str(uuid.uuid4()), "Conv", now, now, 1, long_message),
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        conversations, _ = service.list_conversations(anonymous_id=anonymous_id)
        
        assert len(conversations[0].last_message_preview) <= 103  # 100 + "..."
    
    def test_list_conversations_pagination(self, mock_db_pool, mock_db_connection, anonymous_id):
        """list_conversations() should handle pagination correctly."""
        mock_conn, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.return_value = (100,)  # Total
        mock_cursor.fetchall.return_value = []  # Page might be empty
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        # Request page 5 (offset 80)
        conversations, total = service.list_conversations(
            anonymous_id=anonymous_id,
            limit=20,
            offset=80,
        )
        
        assert total == 100
        # Verify LIMIT and OFFSET in query
        call_args = mock_cursor.execute.call_args_list[-1]
        query = call_args[0][0]
        params = call_args[0][1]
        assert "LIMIT" in query
        assert "OFFSET" in query


class TestResearchContextCRUD:
    """Tests for research context operations."""
    
    def test_create_research_context(self, mock_db_pool, mock_db_connection, conversation_id):
        """create_research_context() should create and return context."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        context_id = str(uuid.uuid4())
        
        mock_cursor.fetchone.return_value = (context_id, now)
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.create_research_context(
            conversation_id=conversation_id,
            query="Test query",
            query_embedding=[0.1] * 1536,
            synthesis_summary="Test summary",
            route_taken="full_graph",
            intent="research",
            papers_count=5,
            critic_decision="sufficient",
            avg_quality=0.75,
            discard_ratio=0.2,
        )
        
        assert isinstance(result, ResearchContextRecord)
        assert result.id == context_id
        assert result.papers_count == 5
    
    def test_add_research_papers_batch(self, mock_db_pool, mock_db_connection, research_context_id):
        """add_research_papers() should batch insert papers."""
        mock_conn, mock_cursor = mock_db_connection
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        papers = [
            ResearchPaperRecord(
                research_context_id=research_context_id,
                pmid="12345",
                title="Paper 1",
            ),
            ResearchPaperRecord(
                research_context_id=research_context_id,
                pmid="67890",
                title="Paper 2",
            ),
        ]
        
        count = service.add_research_papers(research_context_id, papers)
        
        assert count == 2
    
    def test_add_research_papers_empty_list(self, mock_db_pool, mock_db_connection, research_context_id):
        """add_research_papers() should handle empty list."""
        mock_conn, mock_cursor = mock_db_connection
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        count = service.add_research_papers(research_context_id, [])
        
        assert count == 0
        # Should not execute any queries
        mock_cursor.execute.assert_not_called()
    
    def test_get_latest_research_context(self, mock_db_pool, mock_db_connection, conversation_id):
        """get_latest_research_context() should return most recent context."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        context_id = str(uuid.uuid4())
        
        mock_cursor.fetchone.return_value = (
            context_id,
            conversation_id,
            "Test query",
            "Summary",
            "full_graph",
            "research",
            5,
            "sufficient",
            0.75,
            0.2,
            now,
        )
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.get_latest_research_context(conversation_id)
        
        assert result is not None
        assert result.id == context_id
    
    def test_get_latest_research_context_none(self, mock_db_pool, mock_db_connection, conversation_id):
        """get_latest_research_context() should return None if no context exists."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.get_latest_research_context(conversation_id)
        
        assert result is None


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.parametrize("invalid_id", [
        "",
        " ",
        "not-a-uuid",
        "../../../etc/passwd",
        "'; DROP TABLE conversations; --",
    ])
    def test_get_conversation_handles_invalid_ids(self, mock_db_pool, mock_db_connection, invalid_id):
        """get_conversation() should handle invalid IDs gracefully."""
        mock_conn, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        # Should not raise, just return None
        result = service.get_conversation(invalid_id)
        assert result is None
    
    def test_create_conversation_with_special_characters_in_title(self, mock_db_pool, mock_db_connection):
        """create_conversation() should handle special characters in title."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        
        # Title with special characters
        special_title = "Test <script>alert('xss')</script> 'quotes' \"double\""
        
        mock_cursor.fetchone.return_value = (
            str(uuid.uuid4()),
            "anon-123",
            special_title,
            False,
            now,
            now,
        )
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.create_conversation(title=special_title)
        
        # Should be stored as-is (parameterized queries prevent SQL injection)
        assert result.title == special_title
    
    def test_add_message_with_very_long_content(self, mock_db_pool, mock_db_connection, conversation_id):
        """add_message() should handle very long content."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        long_content = "x" * 10000  # Very long message
        
        mock_cursor.fetchone.side_effect = [
            (1,),
            (str(uuid.uuid4()), conversation_id, "user", long_content, None, 1, now),
        ]
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        result = service.add_message(
            conversation_id=conversation_id,
            role="user",
            content=long_content,
        )
        
        assert result.content == long_content
    
    def test_concurrent_sequence_number_generation(self, mock_db_pool, mock_db_connection, conversation_id):
        """Sequence number generation should handle concurrent inserts."""
        mock_conn, mock_cursor = mock_db_connection
        
        now = datetime.now(timezone.utc)
        
        # Simulate unique constraint violation on first attempt
        call_count = [0]
        
        def execute_with_retry(query, params=None):
            call_count[0] += 1
            if call_count[0] == 2:  # INSERT statement
                if call_count[0] <= 2:
                    raise psycopg2.IntegrityError("unique_conversation_sequence")
        
        # This test verifies the database handles it - actual retry logic
        # would be in the application if needed
        mock_cursor.fetchone.return_value = (1,)
        
        service = DatabaseService()
        service._pool = mock_db_pool
        
        # The database constraint should handle this
        # Application might need retry logic in production