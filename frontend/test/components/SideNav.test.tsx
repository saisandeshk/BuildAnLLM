import { fireEvent, render, screen } from "@testing-library/react";

import SideNav from "../../components/SideNav";

describe("SideNav", () => {
  it("marks the active section and notifies on navigate", () => {
    const onNavigate = vi.fn();
    const sections = [
      { id: "one", label: "One" },
      { id: "two", label: "Two" },
    ];
    render(<SideNav sections={sections} activeId="two" onNavigate={onNavigate} />);

    const active = screen.getByText("Two");
    expect(active).toHaveAttribute("aria-current", "location");

    fireEvent.click(screen.getByText("One"));
    expect(onNavigate).toHaveBeenCalledWith("one");
  });
});
