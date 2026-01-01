"use client";

import {
  Line,
  LineChart as RechartsLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";

export type LineSeries = {
  dataKey: string;
  name: string;
  color: string;
};

export default function LineChart({
  data,
  lines,
  xKey,
}: {
  data: Array<Record<string, number>>;
  lines: LineSeries[];
  xKey: string;
}) {
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <RechartsLineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <XAxis
            dataKey={xKey}
            stroke="var(--chart-axis)"
            tick={{
              fontSize: 12,
              fill: "var(--chart-text)",
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
            }}
            tickLine={{ stroke: "var(--chart-axis)" }}
            axisLine={{ stroke: "var(--chart-axis)" }}
          />
          <YAxis
            stroke="var(--chart-axis)"
            tick={{
              fontSize: 12,
              fill: "var(--chart-text)",
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
            }}
            tickLine={{ stroke: "var(--chart-axis)" }}
            axisLine={{ stroke: "var(--chart-axis)" }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--tooltip-bg)",
              border: "1px solid var(--stroke-strong)",
              color: "var(--ink-1)",
              fontSize: 12,
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
            }}
            labelStyle={{ color: "var(--ink-1)" }}
            itemStyle={{ color: "var(--ink-1)" }}
          />
          <Legend
            wrapperStyle={{
              color: "var(--chart-text)",
              fontSize: 12,
              fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
            }}
          />
          {lines.map((line) => (
            <Line
              key={line.dataKey}
              type="monotone"
              dataKey={line.dataKey}
              name={line.name}
              stroke={line.color}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}
