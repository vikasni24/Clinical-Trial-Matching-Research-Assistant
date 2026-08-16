import type { ApiError } from "../../api/client";

/** Renders a friendly, user-safe message for any ApiError — never the raw
 * backend exception text. Distinguishes a network/connectivity failure
 * (backend unreachable) from an HTTP error response. */
export function ErrorBlock({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  const title = error.kind === "network" ? "Backend unavailable" : "Something went wrong";

  return (
    <div className="error-block">
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <div>{error.message}</div>
      {onRetry && (
        <button type="button" className="btn btn-secondary" style={{ marginTop: 10 }} onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
