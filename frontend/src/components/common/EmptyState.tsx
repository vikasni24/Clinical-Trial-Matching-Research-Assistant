import type { ReactNode } from "react";

/** A single, explicit empty-state component used throughout the app.
 * Callers must pass a specific title/hint distinguishing WHY there's
 * nothing to show (no data exists vs. no match vs. insufficient evidence
 * vs. unsupported query) — these are deliberately not collapsed into one
 * generic "Nothing here" message anywhere in this app. */
export function EmptyBlock({ title, hint, icon }: { title: string; hint?: string; icon?: ReactNode }) {
  return (
    <div className="state-block">
      {icon}
      <div className="state-block-title">{title}</div>
      {hint && <p className="state-block-hint">{hint}</p>}
    </div>
  );
}
