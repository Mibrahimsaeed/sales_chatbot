const API_URL = import.meta.env.VITE_API_URL || "http://172.16.8.193:8000/api";

let cachedToken = null;

/**
 * DEV ONLY: obtains a token from the throwaway /token endpoint.
 * Replace this later with the dashboard's real auth token.
 */
async function getDevToken() {
  if (cachedToken) return cachedToken;

  console.log("Getting dev token...");

  const res = await fetch(`${API_URL}/token`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      name: "Dashboard User",
    }),
  });

  console.log("Token endpoint status:", res.status);

  if (!res.ok) {
    const bodyText = await res.text().catch(() => "");
    console.error("Failed to obtain dev token:", bodyText);
    throw new Error(`Failed to obtain dev token: ${res.status}`);
  }

  const data = await res.json();
  cachedToken = data.access_token;

  console.log("Token received successfully.");

  return cachedToken;
}

async function getCurrentToken() {
  return getDevToken();
}

export async function sendChatMessage(message, sessionId) {
  console.log("Calling backend...");
  console.log("Message:", message);

  const token = await getCurrentToken();

  const res = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  });

  console.log("Chat endpoint status:", res.status);

  if (!res.ok) {
    const bodyText = await res.text().catch(() => "");
    console.error("Chat request failed:", bodyText);
    throw new Error(`Chat request failed: ${res.status}`);
  }

  return await res.json();
}

/**
 * Part 8 (pagination): fetches the next page of a result set that's
 * already been shown, using the session's server-side pagination cursor
 * — no message text, no re-query/re-filter. Same response shape as
 * sendChatMessage, but `data` here is only the NEW batch; the caller
 * appends it to what it already has.
 */
export async function fetchMoreResults(sessionId) {
  const token = await getCurrentToken();

  const res = await fetch(`${API_URL}/chat/more`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      session_id: sessionId,
    }),
  });

  if (!res.ok) {
    const bodyText = await res.text().catch(() => "");
    console.error("Show-more request failed:", bodyText);
    throw new Error(`Show-more request failed: ${res.status}`);
  }

  return await res.json();
}