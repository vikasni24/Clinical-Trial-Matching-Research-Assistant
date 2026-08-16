import type { Evidence } from "../types/api";
import type { TrendPoint } from "../components/charts/ObservationTrendChart";

export interface ObservationTrend {
  label: string;
  unit: string | null;
  points: TrendPoint[];
}

/**
 * Groups a patient's Observation Evidence items by code (falling back to
 * display text) and keeps only numeric-valued groups with more than one
 * distinct dated reading — i.e. genuine trends. Every point is a real,
 * already-returned Evidence item's own effective_date/value; nothing is
 * interpolated, estimated, or invented. Groups are sorted so the series
 * with the most data points (the most clinically "trackable" one) appears
 * first.
 */
export function groupObservationTrends(evidence: Evidence[]): ObservationTrend[] {
  const groups = new Map<string, { label: string; unit: string | null; points: TrendPoint[] }>();

  for (const item of evidence) {
    if (typeof item.value !== "number") continue;
    if (!item.effective_date) continue;

    const key = item.code ?? item.display ?? item.resource_type;
    const label = item.display ?? item.code ?? "Observation";
    const existing = groups.get(key);
    const point: TrendPoint = { date: item.effective_date.slice(0, 10), value: item.value };

    if (existing) {
      existing.points.push(point);
    } else {
      groups.set(key, { label, unit: item.unit, points: [point] });
    }
  }

  return Array.from(groups.values())
    .filter((g) => g.points.length > 1)
    .map((g) => ({ ...g, points: g.points.sort((a, b) => a.date.localeCompare(b.date)) }))
    .sort((a, b) => b.points.length - a.points.length);
}
