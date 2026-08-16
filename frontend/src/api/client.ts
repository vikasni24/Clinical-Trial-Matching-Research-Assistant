/**
 * Centralized API client. Every backend call in this application goes
 * through `request()` below — components and page code never call
 * fetch/axios directly. Handles the base URL, JSON parsing, HTTP errors,
 * and network failures with one consistent error representation
 * (ApiError), so every page can render error states uniformly.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ?? "";

if (!BASE_URL) {
  // Fails loudly at build/dev time rather than silently hitting a
  // same-origin relative path that would never reach the backend.
  // eslint-disable-next-line no-console
  console.warn(
    "VITE_API_BASE_URL is not set — API requests will be sent to a relative path. " +
      "Set it in frontend/.env (see .env.example)."
  );
}

export type ApiErrorKind = "network" | "http";

/** A single, consistent error shape for every failure mode the UI needs to
 * distinguish: a network/connectivity failure (backend unreachable) vs. an
 * HTTP error response (4xx/5xx) with its status code, never exposing the
 * raw backend exception text to the user. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  searchParams?: Record<string, string | number | undefined>;
}

function buildUrl(path: string, searchParams?: RequestOptions["searchParams"]): string {
  const url = new URL(BASE_URL + path, BASE_URL ? undefined : window.location.origin);
  if (searchParams) {
    for (const [key, value] of Object.entries(searchParams)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

/** Fallback messages per status code, used only when the response carries
 * no usable `detail` string. The backend deliberately curates every
 * `detail` it returns to be safe to show a user (static, secret-free
 * configuration/validation messages, or an intentionally generic message
 * for anything that could otherwise leak upstream provider/database
 * internals — see app/api/routes/patients.py's own error handling and the
 * Phase 10I security hardening pass). Because the backend already does
 * that curation, this client trusts and displays `detail` directly rather
 * than re-genericizing it — the previous behavior discarded the backend's
 * own actionable message (e.g. "LLM provider is not configured: ...") in
 * favor of an unhelpful generic string. */
function fallbackMessageForStatus(status: number): string {
  switch (status) {
    case 400:
      return "That request wasn't valid. Please check your input and try again.";
    case 404:
      return "The requested item could not be found.";
    case 422:
      return "The request was missing required information.";
    case 500:
      return "The server is temporarily unable to process this request.";
    case 502:
      return "The research assistant's language model provider is temporarily unavailable.";
    case 503:
      return "The backend database is temporarily unavailable. Please try again shortly.";
    default:
      return `The request failed (status ${status}).`;
  }
}

async function extractDetail(response: Response): Promise<string | null> {
  try {
    const body: unknown = await response.clone().json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail.trim()) return detail;
    }
  } catch {
    // Not JSON, or no body — fall through to the generic message.
  }
  return null;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, searchParams } = options;
  const url = buildUrl(path, searchParams);

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError("network", "Could not reach the backend. Please check your connection and try again.");
  }

  if (!response.ok) {
    const detail = await extractDetail(response);
    throw new ApiError("http", detail ?? fallbackMessageForStatus(response.status), response.status);
  }

  // No-content responses (not currently used, but handled defensively).
  if (response.status === 204) {
    return undefined as T;
  }

  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("network", "The server returned an unexpected response.");
  }
}

export function getJson<T>(path: string, searchParams?: RequestOptions["searchParams"]): Promise<T> {
  return request<T>(path, { method: "GET", searchParams });
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body });
}
