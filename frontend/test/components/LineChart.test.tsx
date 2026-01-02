import React from "react";
import { render, screen } from "@testing-library/react";

vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="responsive">{children}</div>
  ),
  LineChart: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="line-chart">{children}</div>
  ),
  Line: ({ dataKey }: { dataKey: string }) => <div data-testid={`line-${dataKey}`} />,
  XAxis: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="x-axis">{children}</div>
  ),
  YAxis: ({ children }: { children?: React.ReactNode }) => (
    <div data-testid="y-axis">{children}</div>
  ),
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  Label: ({ value }: { value?: string }) => <span>{value}</span>,
}));

import LineChart from "../../components/LineChart";

describe("LineChart", () => {
  it("renders line series and labels", () => {
    render(
      <LineChart
        data={[{ step: 1, loss: 0.2 }]}
        lines={[{ dataKey: "loss", name: "Loss", color: "#f00" }]}
        xKey="step"
        xLabel="Step"
        yLabel="Loss"
      />
    );

    expect(screen.getByTestId("line-loss")).toBeInTheDocument();
    expect(screen.getByText("Step")).toBeInTheDocument();
    expect(screen.getByText("Loss")).toBeInTheDocument();
  });
});
