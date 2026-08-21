import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DarkTooltip } from "./ChartTooltip";
import { chartColors } from "../../utils/chartTheme";

export interface TrialPhaseDatum {
  phase: string;
  count: number;
}

export function TrialPhaseChart({ data }: { data: TrialPhaseDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid stroke={chartColors.grid} vertical={false} />
        <XAxis dataKey="phase" tick={{ fill: chartColors.textMuted, fontSize: 11.5 }} axisLine={{ stroke: chartColors.grid }} tickLine={false} />
        <YAxis allowDecimals={false} tick={{ fill: chartColors.textMuted, fontSize: 11.5 }} axisLine={false} tickLine={false} />
        <Tooltip content={<DarkTooltip />} cursor={{ fill: "rgba(46,230,166,0.06)" }} />
        <Bar dataKey="count" name="Trials" fill={chartColors.green} radius={[6, 6, 0, 0]} maxBarSize={44} />
      </BarChart>
    </ResponsiveContainer>
  );
}
