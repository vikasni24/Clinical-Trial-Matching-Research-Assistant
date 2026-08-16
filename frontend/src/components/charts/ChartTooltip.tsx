interface TooltipPayloadItem {
  name?: string;
  value?: number | string;
  color?: string;
}

export function DarkTooltip({ active, payload, label }: { active?: boolean; payload?: TooltipPayloadItem[]; label?: string }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div
      style={{
        background: "#121826",
        border: "1px solid #2a3348",
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: 12.5,
        boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
      }}
    >
      {label && <div style={{ color: "#97a1b8", marginBottom: 4 }}>{label}</div>}
      {payload.map((item, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, color: "#eef1f7" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: item.color, display: "inline-block" }} />
          <span>{item.name}: </span>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}
