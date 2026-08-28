import type {
  AnalyzeResponse,
  ExplainableAttribute,
  ExplainResponse,
  VideoAnalyzeResponse,
} from "./types";

// Trailing slashes stripped: "https://api.example.com/" + "/api/..." would
// produce a double-slash path that routes to a 404.
export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export const WS_BASE = API_BASE.replace(/^http/, "ws");

async function throwIfNotOk(res: Response): Promise<void> {
  if (res.ok) return;
  let detail = res.statusText;
  try {
    detail = (await res.json()).detail ?? detail;
  } catch {
    /* non-JSON error body */
  }
  throw new Error(detail);
}

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_BASE}/api/analyze/image`, {
    method: "POST",
    body,
  });
  await throwIfNotOk(res);
  return res.json();
}

export async function analyzeVideo(file: File): Promise<VideoAnalyzeResponse> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_BASE}/api/analyze/video`, {
    method: "POST",
    body,
  });
  await throwIfNotOk(res);
  return res.json();
}

export async function explainAttribute(
  imageBase64: string,
  attribute: ExplainableAttribute,
): Promise<ExplainResponse> {
  const res = await fetch(`${API_BASE}/api/analyze/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageBase64, attribute }),
  });
  await throwIfNotOk(res);
  return res.json();
}

export function openRealtimeSocket(): WebSocket {
  return new WebSocket(`${WS_BASE}/ws/realtime`);
}
