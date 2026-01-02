import { render } from "@testing-library/react";

import TokenRainbow from "../../components/TokenRainbow";

describe("TokenRainbow", () => {
  it("renders colored token spans", () => {
    const { container } = render(<TokenRainbow tokens={["a", "b"]} />);
    const spans = container.querySelectorAll("span");
    expect(spans.length).toBe(2);
    const background = spans[0].style.backgroundColor || spans[0].style.background;
    expect(background).not.toBe("");
    expect(background).toMatch(/(hsl|rgb)\(/);
  });
});
