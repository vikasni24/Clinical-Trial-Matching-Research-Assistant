import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DarkTooltip } from "./ChartTooltip";
import { chartColors } from "../../utils/chartTheme";

export interface EvidenceCategoryDatum {
  resourceType: string;
  count: number;
}

/** Real, per-patient evidence counts by resource type — each count comes
 * from GET /api/patients/{id}/evidence?resource_type=X's own
 * pagination.total, never fabricated or estimated. */
export function EvidenceCategoryChart({ data }: { data: EvidenceCategoryDatum[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid stroke={chartColors.grid} horizontal={false} />
        <XAxis type="number" allowDecimals={false} tick={{ fill: chartColors.textMuted, fontSize: 11.5 }} axisLine={false} tickLine={false} />
        <YAxis
          type="category"
          dataKey="resourceType"
          width={110}
          tick={{ fill: chartColors.textMuted, fontSize: 11.5 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<DarkTooltip />} cursor={{ fill: "rgba(46,230,166,0.06)" }} />
        <Bar dataKey="count" name="Evidence items" fill={chartColors.green} radius={[0, 6, 6, 0]} maxBarSize={18} />
      </BarChart>
    </ResponsiveContainer>
  );
}
