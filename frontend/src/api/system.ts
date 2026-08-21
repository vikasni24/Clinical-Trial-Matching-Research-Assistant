import { getJson } from "./client";

export interface HealthOut {
  status: string;
  /** Whether the backend has a working LLM provider configured. Absent
   * on older backend versions that predate this field. */
  llm_configured?: boolean;
  /** The configured provider's name (e.g. "groq", "anthropic") — never a
   * secret, just which vendor is in use. */
  llm_provider?: string;
}

export function getHealth(): Promise<HealthOut> {
  return getJson<HealthOut>("/health");
}
