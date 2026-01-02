import { render } from "@testing-library/react";

import TokenSegments from "../../components/TokenSegments";

describe("TokenSegments", () => {
  it("renders tokens with fallback for empty strings", () => {
    const { container } = render(
      <TokenSegments tokens={["hi", "", "there"]} tone="prompt" />
    );
    const spans = container.querySelectorAll("span");
    expect(spans.length).toBe(3);
    expect(spans[1].textContent).toBe(" ");
  });
});
