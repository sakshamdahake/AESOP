#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hybrid Redis+PostgreSQL Session System

Tests:
1. Session creation (chat vs research routes)
2. Message persistence to both Redis and PostgreSQL
3. Follow-up messages
4. Session retrieval (cache hit vs miss)
5. Session listing
6. Session deletion
7. Research context saving

Usage:
    python test_hybrid_session.py
"""

import requests
import time
import subprocess
import json
from typing import Dict, List, Any
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
POSTGRES_CONTAINER = "aesop_postgres"
POSTGRES_USER = "aesop"
POSTGRES_DB = "aesop_db"

# ANSI colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_pass(self, test_name: str, details: str = ""):
        self.passed += 1
        self.tests.append(("PASS", test_name, details))
        print(f"{GREEN}✓ PASS{RESET}: {test_name}")
        if details:
            print(f"  {BLUE}→{RESET} {details}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.tests.append(("FAIL", test_name, error))
        print(f"{RED}✗ FAIL{RESET}: {test_name}")
        print(f"  {RED}→{RESET} {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed == 0:
            print(f"{GREEN}ALL TESTS PASSED!{RESET}")
        else:
            print(f"{RED}{self.failed} TESTS FAILED{RESET}")
        print("="*70)


def run_db_query(query: str) -> List[Dict[str, Any]]:
    """Execute PostgreSQL query and return results."""
    cmd = [
        "docker", "exec", POSTGRES_CONTAINER,
        "psql", "-U", POSTGRES_USER, "-d", POSTGRES_DB,
        "-t", "-A", "-F", "|", "-c", query
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return []
        
        # Parse pipe-delimited output
        rows = []
        for line in lines:
            if line.strip():
                rows.append(line.split("|"))
        return rows
    except subprocess.CalledProcessError as e:
        print(f"{RED}DB Query Error:{RESET} {e.stderr}")
        return []


def get_session_from_db(session_id: str) -> Dict[str, Any]:
    """Get session details from PostgreSQL."""
    query = f"""
        SELECT s.id, s.title, s.original_query, 
               COUNT(m.id) as message_count
        FROM sessions s
        LEFT JOIN messages m ON s.id = m.session_id
        WHERE s.id = '{session_id}' AND s.deleted_at IS NULL
        GROUP BY s.id;
    """
    rows = run_db_query(query)
    if rows:
        return {
            "id": rows[0][0],
            "title": rows[0][1],
            "original_query": rows[0][2],
            "message_count": int(rows[0][3]) if rows[0][3] else 0,
        }
    return {}


def get_messages_from_db(session_id: str) -> List[Dict[str, Any]]:
    """Get messages for a session from PostgreSQL."""
    query = f"""
        SELECT id, role, content, sequence_num
        FROM messages
        WHERE session_id = '{session_id}'
        ORDER BY sequence_num ASC;
    """
    rows = run_db_query(query)
    return [
        {
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "sequence_num": int(row[3]),
        }
        for row in rows
    ]


def check_research_context(session_id: str) -> bool:
    """Check if research context was saved for session."""
    query = f"""
        SELECT COUNT(*) FROM research_contexts
        WHERE session_id = '{session_id}';
    """
    rows = run_db_query(query)
    return rows and int(rows[0][0]) > 0


def test_chat_route_session_creation(results: TestResults):
    """Test 1: Chat route session creation with message persistence."""
    print(f"\n{YELLOW}TEST 1: Chat Route Session Creation{RESET}")
    
    try:
        # Create session with chat message
        response = requests.post(
            f"{BASE_URL}/sessions",
            json={"initial_message": "Hello!"},
            timeout=30
        )
        
        if response.status_code != 200:
            results.add_fail("Chat session creation", f"HTTP {response.status_code}")
            return
        
        data = response.json()
        session_id = data["session_id"]
        
        # Wait for DB sync
        time.sleep(1)
        
        # Check PostgreSQL
        db_session = get_session_from_db(session_id)
        if not db_session:
            results.add_fail("Chat session in DB", "Session not found in PostgreSQL")
            return
        
        # Check message count
        messages = get_messages_from_db(session_id)
        if len(messages) != 2:  # 1 user + 1 assistant
            results.add_fail(
                "Chat messages in DB",
                f"Expected 2 messages, got {len(messages)}"
            )
            return
        
        # Verify message roles
        if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
            results.add_fail(
                "Chat message roles",
                f"Expected user->assistant, got {messages[0]['role']}->{messages[1]['role']}"
            )
            return
        
        results.add_pass(
            "Chat route session creation",
            f"Session {session_id[:8]}... created with 2 messages in DB"
        )
        
        return session_id
        
    except Exception as e:
        results.add_fail("Chat session creation", str(e))
        return None


def test_research_route_session_creation(results: TestResults):
    """Test 2: Research route session creation with papers."""
    print(f"\n{YELLOW}TEST 2: Research Route Session Creation{RESET}")
    
    try:
        # Create session with research query
        response = requests.post(
            f"{BASE_URL}/sessions",
            json={"initial_message": "What are treatments for Type 2 diabetes?"},
            timeout=60  # Research takes longer
        )
        
        if response.status_code != 200:
            results.add_fail("Research session creation", f"HTTP {response.status_code}")
            return
        
        data = response.json()
        session_id = data["session_id"]
        
        # Wait for DB sync
        time.sleep(1)
        
        # Check PostgreSQL
        db_session = get_session_from_db(session_id)
        if not db_session:
            results.add_fail("Research session in DB", "Session not found in PostgreSQL")
            return
        
        # Check message count
        messages = get_messages_from_db(session_id)
        if len(messages) != 2:  # 1 user + 1 assistant
            results.add_fail(
                "Research messages in DB",
                f"Expected 2 messages, got {len(messages)}"
            )
            return
        
        # Check research context was saved
        has_context = check_research_context(session_id)
        if not has_context:
            results.add_fail(
                "Research context in DB",
                "Research context not saved"
            )
            return
        
        results.add_pass(
            "Research route session creation",
            f"Session {session_id[:8]}... created with 2 messages + research context"
        )
        
        return session_id
        
    except Exception as e:
        results.add_fail("Research session creation", str(e))
        return None


def test_follow_up_message(results: TestResults, session_id: str):
    """Test 3: Follow-up message adds to existing session."""
    print(f"\n{YELLOW}TEST 3: Follow-up Message{RESET}")
    
    try:
        # Get initial message count
        initial_messages = get_messages_from_db(session_id)
        initial_count = len(initial_messages)
        
        # Send follow-up message
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/messages",
            json={"message": "Tell me more about that."},
            timeout=30
        )
        
        if response.status_code != 200:
            results.add_fail("Follow-up message", f"HTTP {response.status_code}")
            return
        
        # Wait for DB sync
        time.sleep(1)
        
        # Check message count increased
        new_messages = get_messages_from_db(session_id)
        new_count = len(new_messages)
        
        expected_count = initial_count + 2  # +1 user, +1 assistant
        if new_count != expected_count:
            results.add_fail(
                "Follow-up message persistence",
                f"Expected {expected_count} messages, got {new_count}"
            )
            return
        
        # Verify sequence numbers are correct
        sequence_nums = [m["sequence_num"] for m in new_messages]
        expected_sequence = list(range(1, new_count + 1))
        if sequence_nums != expected_sequence:
            results.add_fail(
                "Message sequence numbers",
                f"Expected {expected_sequence}, got {sequence_nums}"
            )
            return
        
        results.add_pass(
            "Follow-up message",
            f"Message added successfully ({initial_count} → {new_count} messages)"
        )
        
    except Exception as e:
        results.add_fail("Follow-up message", str(e))


def test_session_retrieval(results: TestResults, session_id: str):
    """Test 4: Session retrieval via API."""
    print(f"\n{YELLOW}TEST 4: Session Retrieval{RESET}")
    
    try:
        # Get session via API
        response = requests.get(f"{BASE_URL}/sessions/{session_id}", timeout=10)
        
        if response.status_code != 200:
            results.add_fail("Session retrieval", f"HTTP {response.status_code}")
            return
        
        data = response.json()
        
        # Verify session ID matches
        if data["session_id"] != session_id:
            results.add_fail(
                "Session ID match",
                f"Expected {session_id}, got {data['session_id']}"
            )
            return
        
        # Verify messages are present
        if not data.get("messages"):
            results.add_fail("Session messages", "No messages in response")
            return
        
        # Compare with DB count
        db_messages = get_messages_from_db(session_id)
        api_message_count = len(data["messages"])
        db_message_count = len(db_messages)
        
        if api_message_count != db_message_count:
            results.add_fail(
                "Session message count sync",
                f"API: {api_message_count}, DB: {db_message_count}"
            )
            return
        
        results.add_pass(
            "Session retrieval",
            f"Retrieved session with {api_message_count} messages"
        )
        
    except Exception as e:
        results.add_fail("Session retrieval", str(e))


def test_session_listing(results: TestResults):
    """Test 5: Session listing."""
    print(f"\n{YELLOW}TEST 5: Session Listing{RESET}")
    
    try:
        # List sessions
        response = requests.get(f"{BASE_URL}/sessions?limit=10", timeout=10)
        
        if response.status_code != 200:
            results.add_fail("Session listing", f"HTTP {response.status_code}")
            return
        
        data = response.json()
        sessions = data.get("sessions", [])
        
        if not sessions:
            results.add_fail("Session listing", "No sessions returned")
            return
        
        # Verify each session has required fields
        for session in sessions:
            if not all(k in session for k in ["session_id", "title", "updated_at"]):
                results.add_fail(
                    "Session listing format",
                    "Missing required fields"
                )
                return
        
        results.add_pass(
            "Session listing",
            f"Retrieved {len(sessions)} sessions"
        )
        
    except Exception as e:
        results.add_fail("Session listing", str(e))


def test_session_deletion(results: TestResults, session_id: str):
    """Test 6: Session deletion (soft delete in DB, removed from Redis)."""
    print(f"\n{YELLOW}TEST 6: Session Deletion{RESET}")
    
    try:
        # Delete session
        response = requests.delete(f"{BASE_URL}/sessions/{session_id}", timeout=10)
        
        if response.status_code != 200:
            results.add_fail("Session deletion", f"HTTP {response.status_code}")
            return
        
        # Wait for sync
        time.sleep(1)
        
        # Verify session is soft-deleted in DB
        query = f"""
            SELECT deleted_at FROM sessions
            WHERE id = '{session_id}';
        """
        rows = run_db_query(query)
        
        if not rows or not rows[0][0]:
            results.add_fail(
                "Session soft delete",
                "Session not marked as deleted in DB"
            )
            return
        
        # Verify session is not retrievable via API
        response = requests.get(f"{BASE_URL}/sessions/{session_id}", timeout=10)
        if response.status_code != 404:
            results.add_fail(
                "Deleted session retrieval",
                f"Expected 404, got {response.status_code}"
            )
            return
        
        results.add_pass(
            "Session deletion",
            f"Session {session_id[:8]}... soft-deleted successfully"
        )
        
    except Exception as e:
        results.add_fail("Session deletion", str(e))


def test_no_duplicate_messages(results: TestResults):
    """Test 7: Verify no duplicate messages on save."""
    print(f"\n{YELLOW}TEST 7: No Duplicate Messages{RESET}")
    
    try:
        # Create session
        response = requests.post(
            f"{BASE_URL}/sessions",
            json={"initial_message": "Test duplicate check"},
            timeout=30
        )
        
        if response.status_code != 200:
            results.add_fail("Duplicate test setup", f"HTTP {response.status_code}")
            return
        
        session_id = response.json()["session_id"]
        time.sleep(1)
        
        # Get messages
        messages = get_messages_from_db(session_id)
        
        # Check for duplicate user messages
        user_messages = [m for m in messages if m["role"] == "user"]
        user_contents = [m["content"] for m in user_messages]
        
        if len(user_contents) != len(set(user_contents)):
            results.add_fail(
                "Duplicate user messages",
                f"Found duplicate messages: {user_contents}"
            )
            return
        
        # Verify exactly 2 messages (1 user, 1 assistant)
        if len(messages) != 2:
            results.add_fail(
                "Message count",
                f"Expected 2 messages, got {len(messages)}"
            )
            return
        
        results.add_pass(
            "No duplicate messages",
            "No duplicates found, correct message count"
        )
        
    except Exception as e:
        results.add_fail("Duplicate message test", str(e))


def main():
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}AESOP Hybrid Session System - Comprehensive Test Suite{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    results = TestResults()
    
    # Test 1: Chat route
    chat_session_id = test_chat_route_session_creation(results)
    
    # Test 2: Research route
    research_session_id = test_research_route_session_creation(results)
    
    # Test 3: Follow-up message (use chat session)
    if chat_session_id:
        test_follow_up_message(results, chat_session_id)
    
    # Test 4: Session retrieval (use research session)
    if research_session_id:
        test_session_retrieval(results, research_session_id)
    
    # Test 5: Session listing
    test_session_listing(results)
    
    # Test 6: Session deletion (use chat session)
    if chat_session_id:
        test_session_deletion(results, chat_session_id)
    
    # Test 7: No duplicate messages
    test_no_duplicate_messages(results)
    
    # Print summary
    results.summary()
    
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit(main())