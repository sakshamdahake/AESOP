import {
  useParams,
  useNavigate,
  useOutletContext,
} from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import type { OutletContextType } from "../../layouts/MainLayout";

import EmptyState from "../../components/workspace/EmptyState";
import ChatInput from "../../components/workspace/ChatInput";
import AnswerBlock from "../../components/workspace/AnswerBlock";

import { createSession, getSession } from "../../api/sessions";
import { streamMessage } from "../../api/stream";

import type {
  AnswerSection,
  MessageMetadata,
  StreamEvent,
  SessionMessage,
} from "../../api/types";

type UIMessage =
  | {
      role: "user";
      content: string;
      timestamp?: string;
    }
  | {
      role: "assistant";
      sections: {
        type: AnswerSection["type"];
        content: string;
        streaming?: boolean;
      }[];
      metadata?: MessageMetadata;
      timestamp?: string;
    };

function Workspace() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  // 🔗 Correct communication with MainLayout
  const { refreshSessions } =
    useOutletContext<OutletContextType>();

  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  // ─────────────────────────────────────────────
  // Load existing session on /chat/:sessionId
  // ─────────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      return;
    }

    const load = async () => {
      setInitialLoading(true);
      const res = await getSession(sessionId);

      const hydrated: UIMessage[] = res.messages.map(
        (m: SessionMessage) => {
          if (m.role === "user") {
            return {
              role: "user",
              content: m.content ?? "",
              timestamp: m.timestamp,
            };
          }

          return {
            role: "assistant",
            sections:
              m.answer?.sections.map((s) => ({
                type: s.type,
                content: s.content,
              })) ?? [],
            metadata: m.metadata,
            timestamp: m.timestamp,
          };
        }
      );

      setMessages(hydrated);
      setInitialLoading(false);
    };

    load();
  }, [sessionId]);

  // ─────────────────────────────────────────────
  // Send message (create or stream)
  // ─────────────────────────────────────────────
  const handleSend = async (text: string) => {
    if (!text.trim()) return;

    // Cancel any ongoing stream
    abortRef.current?.abort();

    // Add user message immediately
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
    ]);

    // ─────────────────────────────
    // CREATE SESSION (first message)
    // ─────────────────────────────
    if (!sessionId) {
      setLoading(true);

      const res = await createSession(text);

      if (res.initial_response) {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            sections: res.initial_response.answer.sections.map(
              (s) => ({
                type: s.type,
                content: s.content,
              })
            ),
            metadata: res.initial_response.metadata,
          },
        ]);
      }

      // 🔔 Refresh sidebar session list
      refreshSessions();

      navigate(`/chat/${res.session_id}`, { replace: true });
      setLoading(false);
      return;
    }

    // ─────────────────────────────
    // STREAM RESPONSE
    // ─────────────────────────────
    abortRef.current = new AbortController();

    let assistantIndex = -1;

    setMessages((prev) => {
      assistantIndex = prev.length;
      return [
        ...prev,
        {
          role: "assistant",
          sections: [],
        },
      ];
    });

    await streamMessage(
      sessionId,
      text,
      (event: StreamEvent) => {
        setMessages((prev) => {
          const copy = [...prev];
          const msg = copy[assistantIndex];

          if (!msg || msg.role !== "assistant") return prev;

          switch (event.event) {
            case "section_start":
              msg.sections.push({
                type: event.type,
                content: "",
                streaming: true,
              });
              break;

            case "token":
              if (msg.sections.length === 0) {
                msg.sections.push({
                  type: "summary",
                  content: event.content,
                  streaming: true,
                });
              } else {
                msg.sections[msg.sections.length - 1].content +=
                  event.content;
              }
              break;

            case "section_end":
              msg.sections[msg.sections.length - 1].streaming =
                false;
              break;

            case "metadata":
              msg.metadata = event.data;
              break;

            case "error":
              msg.sections.push({
                type: "summary",
                content: `Error: ${event.message}`,
              });
              break;
          }

          return copy;
        });
      },
      abortRef.current.signal
    );
  };

  // ─────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1 }}>
        {initialLoading && (
          <p style={{ color: "#666" }}>Loading session…</p>
        )}

        {!sessionId && messages.length === 0 && <EmptyState />}

        {messages.map((m, idx) => {
          if (m.role === "user") {
            return (
              <div key={idx} style={{ marginBottom: "1rem" }}>
                <strong>You</strong>
                <p>{m.content}</p>
              </div>
            );
          }

          return (
            <AnswerBlock
              key={idx}
              sections={m.sections}
              metadata={m.metadata}
            />
          );
        })}

        {loading && (
          <p style={{ color: "#666" }}>AESOP is thinking…</p>
        )}
      </div>

      <ChatInput onSend={handleSend} />
    </div>
  );
}

export default Workspace;
