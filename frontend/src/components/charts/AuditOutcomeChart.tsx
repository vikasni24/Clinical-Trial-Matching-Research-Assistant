import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DarkTooltip } from "./ChartTooltip";
import { chartColors } from "../../utils/chartTheme";
import type { AnswerStatus } from "../../types/api";

export interface AuditOutcomeDatum {
  status: AnswerStatus;
  label: string;
  count: number;
}

const COLOR_BY_STATUS: Record<AnswerStatus, string> = {
  answered: chartColors.green,
  insufficient_evidence: chartColors.amber,
  unsupported: chartColors.gray,
};

/** Outcome breakdown for the most recently fetched page of this patient's
 * audit history — real counts tallied client-side from already-returned
 * AuditRecordOut items, never a separate aggregate. Labeled "recent" (not
 * "all-time") since only a bounded page is loaded. */
export function AuditOutcomeChart({ data }: { data: AuditOutcomeDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid stroke={chartColors.grid} horizontal={false} />
        <XAxis type="number" allowDecimals={false} tick={{ fill: chartColors.textMuted, fontSize: 11.5 }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="label" width={140} tick={{ fill: chartColors.textMuted, fontSize: 11.5 }} axisLine={false} tickLine={false} />
        <Tooltip content={<DarkTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
        <Bar dataKey="count" name="Questions" radius={[0, 6, 6, 0]} maxBarSize={18}>
          {data.map((d) => (
            <Cell key={d.status} fill={COLOR_BY_STATUS[d.status]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
