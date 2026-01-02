import { fireEvent, render, screen } from "@testing-library/react";

import RangeSlider from "../../components/RangeSlider";

describe("RangeSlider", () => {
  it("renders label and value", () => {
    render(
      <RangeSlider label="Steps" value={3} min={0} max={10} onChange={() => {}} />
    );

    expect(screen.getByText("Steps")).toBeInTheDocument();
    expect(screen.getByText("3 / 10")).toBeInTheDocument();
  });

  it("invokes onChange with numeric value", () => {
    const onChange = vi.fn();
    render(
      <RangeSlider label="Steps" value={3} min={0} max={10} onChange={onChange} />
    );

    const slider = screen.getByRole("slider");
    fireEvent.change(slider, { target: { value: "7" } });
    expect(onChange).toHaveBeenCalledWith(7);
  });
});
