import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { DarkTooltip } from "./ChartTooltip";
import { chartColors } from "../../utils/chartTheme";
import type { OverallEligibilityStatus } from "../../types/api";

export interface EligibilityDonutDatum {
  status: OverallEligibilityStatus;
  count: number;
}

const COLOR_BY_STATUS: Record<OverallEligibilityStatus, string> = {
  ELIGIBLE: chartColors.green,
  INELIGIBLE: chartColors.red,
  UNKNOWN: chartColors.gray,
};

/** Real, per-patient eligibility distribution derived from
 * GET /api/patients/{id}/matches — one count per MatchResultOut's own
 * eligibility_status, never computed or guessed client-side. */
export function EligibilityDonut({ data }: { data: EligibilityDonutDatum[] }) {
  const total = data.reduce((sum, d) => sum + d.count, 0);
  if (total === 0) return null;

  return (
    <div style={{ position: "relative" }}>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Tooltip content={<DarkTooltip />} />
          <Pie data={data} dataKey="count" nameKey="status" innerRadius={58} outerRadius={82} paddingAngle={3} strokeWidth={0}>
            {data.map((d) => (
              <Cell key={d.status} fill={COLOR_BY_STATUS[d.status]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -55%)",
          textAlign: "center",
          pointerEvents: "none",
        }}
      >
        <div style={{ fontSize: 22, fontWeight: 700 }}>{total}</div>
        <div style={{ fontSize: 10.5, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Trials
        </div>
      </div>
    </div>
  );
}
