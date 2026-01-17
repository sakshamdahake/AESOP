import { http } from "./http";
import type {
  CreateSessionResponse,
  ListSessionsResponse,
  SessionDetailResponse,
} from "./types";

// Create session (optional initial message)
export function createSession(
  initialMessage?: string
): Promise<CreateSessionResponse> {
  return http<CreateSessionResponse>("/sessions", {
    method: "POST",
    body: JSON.stringify(
      initialMessage ? { initial_message: initialMessage } : {}
    ),
  });
}

// List sessions (sidebar)
export function listSessions(
  limit = 50,
  offset = 0
): Promise<ListSessionsResponse> {
  return http<ListSessionsResponse>(
    `/sessions?limit=${limit}&offset=${offset}`
  );
}

// Get session details + messages
export function getSession(
  sessionId: string
): Promise<SessionDetailResponse> {
  return http<SessionDetailResponse>(`/sessions/${sessionId}`);
}

// Delete session
export function deleteSession(sessionId: string): Promise<{
  status: string;
  session_id: string;
}> {
  return http(`/sessions/${sessionId}`, {
    method: "DELETE",
  });
}
