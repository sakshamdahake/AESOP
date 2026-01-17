import { http } from "./http";
import type { MessageResponse } from "./types";

export function sendMessage(
  sessionId: string,
  message: string
): Promise<MessageResponse> {
  return http<MessageResponse>(`/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ message }),
  });
}
