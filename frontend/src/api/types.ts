// ===== Answer / Sections =====

export type AnswerSectionType =
  | "summary"
  | "evidence"
  | "methodology"
  | "limitations"
  | "recommendations";

export interface AnswerSection {
  type: AnswerSectionType;
  content: string;
}

export interface StructuredAnswer {
  sections: AnswerSection[];
}

// ===== Metadata =====

export interface MessageMetadata {
  intent: "research" | "followup_research" | "chat" | "utility" | null;
  intent_confidence: number | null;
  processing_route: string;
  papers_count: number;
  review_outcome: "sufficient" | "retrieve_more" | null;
  evidence_score: number | null;
}

// ===== Messages =====

export interface SessionMessage {
  role: "user" | "assistant";
  content?: string; // user
  answer?: StructuredAnswer; // assistant
  metadata?: MessageMetadata;
  timestamp: string;
}

// ===== Sessions =====

export interface SessionSummary {
  session_id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

export interface SessionDetailResponse {
  session_id: string;
  title: string;
  messages: SessionMessage[];
  papers_count: number;
  created_at: string;
  updated_at: string;
}

// ===== Requests / Responses =====

export interface CreateSessionResponse {
  session_id: string;
  created_at: string;
  title: string | null;
  initial_response: MessageResponse | null;
}

export interface MessageResponse {
  answer: StructuredAnswer;
  metadata: MessageMetadata;
}

export interface ListSessionsResponse {
  sessions: SessionSummary[];
}

// ===== Streaming =====

export type StreamEvent =
  | { event: "section_start"; type: AnswerSectionType }
  | { event: "token"; content: string }
  | { event: "section_end" }
  | { event: "metadata"; data: MessageMetadata }
  | { event: "error"; message: string; code?: string };
