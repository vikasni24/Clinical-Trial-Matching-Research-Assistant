import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DarkTooltip } from "./ChartTooltip";
import { chartColors } from "../../utils/chartTheme";

export interface TrialStatusDatum {
  status: string;
  count: number;
}

/** Real, backend-derived trial status distribution — each count comes
 * from GET /api/trials?status=X's own pagination.total, never fabricated. */
export function TrialStatusChart({ data }: { data: TrialStatusDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid stroke={chartColors.grid} vertical={false} />
        <XAxis dataKey="status" tick={{ fill: chartColors.textMuted, fontSize: 11.5 }} axisLine={{ stroke: chartColors.grid }} tickLine={false} />
        <YAxis allowDecimals={false} tick={{ fill: chartColors.textMuted, fontSize: 11.5 }} axisLine={false} tickLine={false} />
        <Tooltip content={<DarkTooltip />} cursor={{ fill: "rgba(59,130,246,0.06)" }} />
        <Bar dataKey="count" name="Trials" fill={chartColors.blue} radius={[6, 6, 0, 0]} maxBarSize={44} />
      </BarChart>
    </ResponsiveContainer>
  );
}
