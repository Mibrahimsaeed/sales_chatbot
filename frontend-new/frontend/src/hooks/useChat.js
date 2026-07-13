import { useState, useCallback, useRef } from "react";
import { sendChatMessage } from "../api/client";

export function useChat() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: "bot",
      kind: "text",
      text: "Hi — I'm wired up to live CCMC data: MTD funnel, biometric/login, and performance. Ask me about an advisor, a team, or a company.",
    },
  ]);

  const [isLoading, setIsLoading] = useState(false);

  const sessionId = useRef(`web-${Date.now()}`);
  const nextId = useRef(1);

  const send = useCallback(
    async (text) => {
      console.log("send() called:", text);

      if (!text.trim() || isLoading) return;

      const userMsg = {
        id: nextId.current++,
        role: "user",
        kind: "text",
        text,
      };

      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const response = await sendChatMessage(text, sessionId.current);

        console.log("Backend response:", response);

        setMessages((prev) => [
          ...prev,
          {
            id: nextId.current++,
            role: "bot",
            kind: response.type,
            text: response.reply,
            data: response.data,
            metric: response.metric,
          },
        ]);
      } catch (err) {
        console.error("Chat send failed:", err);

        setMessages((prev) => [
          ...prev,
          {
            id: nextId.current++,
            role: "bot",
            kind: "error",
            text: "Something went wrong reaching the server. Please try again.",
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading]
  );

  return {
    messages,
    isLoading,
    send,
  };
}