import { render, screen } from "@testing-library/react";

import Heatmap from "../../components/Heatmap";

describe("Heatmap", () => {
  it("shows loading placeholder when matrix is empty", () => {
    render(<Heatmap matrix={[]} labels={["a", "b"]} />);
    expect(screen.getByText("Loading attention...")).toBeInTheDocument();
  });
});
