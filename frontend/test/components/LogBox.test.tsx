import { render, screen } from "@testing-library/react";

import LogBox from "../../components/LogBox";

describe("LogBox", () => {
  it("shows default empty text", () => {
    render(<LogBox logs={[]} />);
    expect(screen.getByText("No logs yet.")).toBeInTheDocument();
  });

  it("renders joined logs", () => {
    render(<LogBox logs={["one", "two"]} />);
    const box = screen.getByText(/one/);
    expect(box).toHaveTextContent("one");
    expect(box).toHaveTextContent("two");
  });
});
