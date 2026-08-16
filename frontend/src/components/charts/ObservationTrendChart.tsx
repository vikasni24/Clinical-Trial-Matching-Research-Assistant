import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { DarkTooltip } from "./ChartTooltip";
import { chartColors } from "../../utils/chartTheme";

export interface TrendPoint {
  date: string;
  value: number;
}

/** A trend line for one numeric observation type (e.g. HbA1c), built ONLY
 * from real Evidence items already returned by
 * GET /api/patients/{id}/evidence — every point is one actual stored
 * observation's date + value, never interpolated or invented. */
export function ObservationTrendChart({ data, unit }: { data: TrendPoint[]; unit: string | null }) {
  const valueLabel = unit ? `Value (${unit})` : "Value";
  return (
    <ResponsiveContainer width="100%" height={140}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
        <CartesianGrid stroke={chartColors.grid} vertical={false} />
        <XAxis dataKey="date" tick={{ fill: chartColors.textMuted, fontSize: 10.5 }} axisLine={{ stroke: chartColors.grid }} tickLine={false} />
        <YAxis tick={{ fill: chartColors.textMuted, fontSize: 10.5 }} axisLine={false} tickLine={false} width={36} />
        <Tooltip content={<DarkTooltip />} />
        <Line
          name={valueLabel}
          type="monotone"
          dataKey="value"
          stroke={chartColors.blueBright}
          strokeWidth={2}
          dot={{ r: 3, fill: chartColors.blueBright, strokeWidth: 0 }}
          activeDot={{ r: 5 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
