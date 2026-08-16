import { getJson } from "./client";

export interface HealthOut {
  status: string;
}

export function getHealth(): Promise<HealthOut> {
  return getJson<HealthOut>("/health");
}
