export function Spinner() {
  return <span className="spinner" role="status" aria-label="Loading" />;
}

export function LoadingBlock({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state-block">
      <Spinner />
      <p className="state-block-hint">{label}</p>
    </div>
  );
}

export function SkeletonRows({ rows = 4, height = 18 }: { rows?: number; height?: number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, padding: "4px 0" }}>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton" style={{ height, width: `${85 - i * 6}%` }} />
      ))}
    </div>
  );
}
