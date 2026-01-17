"""
Tests for AESOP API v2.0 endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthCheck:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "operational"


class TestSessionEndpoints:
    """Test /sessions endpoints."""
    
    def test_create_session_empty(self):
        """Create session without initial message."""
        response = client.post("/sessions", json={})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "created_at" in data
        assert data.get("initial_response") is None
    
    def test_create_session_with_message(self):
        """Create session with initial message."""
        response = client.post("/sessions", json={
            "initial_message": "What is aspirin used for?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "title" in data
        assert data.get("initial_response") is not None
        assert "answer" in data["initial_response"]
        assert "sections" in data["initial_response"]["answer"]
    
    def test_list_sessions(self):
        """List sessions returns array."""
        # Create a session first
        client.post("/sessions", json={"initial_message": "Test query"})
        
        response = client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)
    
    def test_list_sessions_pagination(self):
        """Test pagination parameters."""
        response = client.get("/sessions?limit=5&offset=0")
        assert response.status_code == 200
    
    def test_get_session_detail(self):
        """Get session with message history."""
        # Create session
        create_resp = client.post("/sessions", json={
            "initial_message": "What are diabetes treatments?"
        })
        session_id = create_resp.json()["session_id"]
        
        # Get detail
        response = client.get(f"/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "messages" in data
        assert "title" in data
    
    def test_get_session_not_found(self):
        """404 for non-existent session."""
        response = client.get("/sessions/nonexistent-id-12345")
        assert response.status_code == 404
    
    def test_delete_session(self):
        """Delete session."""
        # Create session
        create_resp = client.post("/sessions", json={})
        session_id = create_resp.json()["session_id"]
        
        # Delete
        response = client.delete(f"/sessions/{session_id}")
        assert response.status_code == 200
        
        # Verify deleted
        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 404


class TestMessageEndpoints:
    """Test /sessions/{id}/messages endpoints."""
    
    @pytest.fixture
    def session_id(self):
        """Create a session for message tests."""
        response = client.post("/sessions", json={
            "initial_message": "What are treatments for Type 2 diabetes?"
        })
        return response.json()["session_id"]
    
    def test_send_message(self, session_id):
        """Send message to existing session."""
        response = client.post(f"/sessions/{session_id}/messages", json={
            "message": "What sample sizes did these studies use?"
        })
        assert response.status_code == 200
        data = response.json()
        
        # Check structured answer
        assert "answer" in data
        assert "sections" in data["answer"]
        assert len(data["answer"]["sections"]) > 0
        
        # Check metadata
        assert "metadata" in data
        assert "processing_route" in data["metadata"]
        assert "papers_count" in data["metadata"]
    
    def test_send_message_invalid_session(self):
        """404 for message to non-existent session."""
        response = client.post("/sessions/invalid-session-id/messages", json={
            "message": "Test message"
        })
        assert response.status_code == 404
    
    def test_send_message_empty(self, session_id):
        """Validation error for empty message."""
        response = client.post(f"/sessions/{session_id}/messages", json={
            "message": ""
        })
        assert response.status_code == 422  # Validation error
    
    def test_message_history_persists(self, session_id):
        """Messages are saved to session history."""
        # Send a follow-up
        client.post(f"/sessions/{session_id}/messages", json={
            "message": "Compare these studies"
        })
        
        # Check session detail
        response = client.get(f"/sessions/{session_id}")
        data = response.json()
        
        # Should have: initial user + initial assistant + followup user + followup assistant
        assert len(data["messages"]) >= 4


class TestStreamingEndpoint:
    """Test streaming message endpoint."""
    
    @pytest.fixture
    def session_id(self):
        response = client.post("/sessions", json={
            "initial_message": "What is metformin?"
        })
        return response.json()["session_id"]
    
    def test_stream_returns_ndjson(self, session_id):
        """Streaming endpoint returns NDJSON."""
        response = client.post(
            f"/sessions/{session_id}/messages/stream",
            json={"message": "Summarize this"},
        )
        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers.get("content-type", "")
    
    def test_stream_events_structure(self, session_id):
        """Stream contains expected event types."""
        response = client.post(
            f"/sessions/{session_id}/messages/stream",
            json={"message": "What are the side effects?"},
        )
        
        events = [line for line in response.text.strip().split("\n") if line]
        event_types = set()
        
        for event_line in events:
            import json
            event = json.loads(event_line)
            event_types.add(event.get("event"))
        
        # Should have section_start, token, section_end, metadata
        assert "section_start" in event_types
        assert "token" in event_types
        assert "section_end" in event_types
        assert "metadata" in event_types


class TestLegacyEndpoints:
    """Test backward compatibility of deprecated endpoints."""
    
    def test_legacy_chat_still_works(self):
        """POST /chat continues to work."""
        response = client.post("/chat", json={
            "message": "What is insulin?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "route_taken" in data
    
    def test_legacy_chat_with_session(self):
        """POST /chat with session_id works."""
        # First request
        resp1 = client.post("/chat", json={
            "message": "What are diabetes treatments?"
        })
        session_id = resp1.json()["session_id"]
        
        # Follow-up
        resp2 = client.post("/chat", json={
            "message": "Compare these treatments",
            "session_id": session_id
        })
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id
    
    def test_legacy_session_endpoint_redirect(self):
        """GET /session/{id} redirects to /sessions/{id}."""
        # Create session
        create_resp = client.post("/sessions", json={
            "initial_message": "Test"
        })
        session_id = create_resp.json()["session_id"]
        
        # Use legacy endpoint
        response = client.get(f"/session/{session_id}")
        assert response.status_code == 200
    
    def test_review_simple_stateless(self):
        """POST /review/simple remains stateless."""
        response = client.post("/review/simple?query=What%20is%20aspirin")
        assert response.status_code == 200


class TestStructuredAnswerSchema:
    """Test structured answer format."""
    
    def test_answer_has_sections(self):
        """Answers contain typed sections."""
        response = client.post("/sessions", json={
            "initial_message": "What are the benefits of exercise?"
        })
        data = response.json()
        
        if data.get("initial_response"):
            answer = data["initial_response"]["answer"]
            assert "sections" in answer
            
            for section in answer["sections"]:
                assert "type" in section
                assert "content" in section
                assert section["type"] in [
                    "summary", "evidence", "methodology",
                    "limitations", "recommendations"
                ]


class TestFieldNaming:
    """Test renamed fields per API spec."""
    
    def test_processing_route_not_route_taken(self):
        """Metadata uses processing_route not route_taken."""
        response = client.post("/sessions", json={
            "initial_message": "Test query"
        })
        data = response.json()
        
        if data.get("initial_response"):
            metadata = data["initial_response"]["metadata"]
            assert "processing_route" in metadata
            assert "route_taken" not in metadata
    
    def test_review_outcome_not_critic_decision(self):
        """Metadata uses review_outcome not critic_decision."""
        response = client.post("/sessions", json={
            "initial_message": "What studies exist on aspirin?"
        })
        data = response.json()
        
        if data.get("initial_response"):
            metadata = data["initial_response"]["metadata"]
            assert "review_outcome" in metadata
            assert "critic_decision" not in metadata
    
    def test_evidence_score_not_avg_quality(self):
        """Metadata uses evidence_score not avg_quality."""
        response = client.post("/sessions", json={
            "initial_message": "Review diabetes literature"
        })
        data = response.json()
        
        if data.get("initial_response"):
            metadata = data["initial_response"]["metadata"]
            assert "evidence_score" in metadata
            assert "avg_quality" not in metadata
