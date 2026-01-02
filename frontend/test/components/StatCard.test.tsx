import { render, screen } from "@testing-library/react";

import StatCard from "../../components/StatCard";

describe("StatCard", () => {
  it("renders label and value", () => {
    render(<StatCard label="CPU" value="M1" />);
    expect(screen.getByText("CPU")).toBeInTheDocument();
    expect(screen.getByText("M1")).toBeInTheDocument();
  });
});
