import { useState, useCallback, useRef } from "react";
import { sendChatMessage, fetchMoreResults } from "../api/client";

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
  const [loadingMoreId, setLoadingMoreId] = useState(null);

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
            totalCount: response.total_count,
            hasMore: response.has_more,
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

  // Part 8 (pagination): fetches the next page for a message that has
  // more results, and appends it to that SAME message's data — no new
  // chat bubble, "preserving the existing results" as the spec asks.
  const loadMore = useCallback(
    async (messageId) => {
      if (loadingMoreId) return;
      setLoadingMoreId(messageId);

      try {
        const response = await fetchMoreResults(sessionId.current);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === messageId
              ? {
                  ...m,
                  data: [...(m.data || []), ...(response.data || [])],
                  totalCount: response.total_count,
                  hasMore: response.has_more,
                }
              : m
          )
        );
      } catch (err) {
        console.error("Show more failed:", err);
      } finally {
        setLoadingMoreId(null);
      }
    },
    [loadingMoreId]
  );

  return {
    messages,
    isLoading,
    send,
    loadMore,
    loadingMoreId,
  };
}