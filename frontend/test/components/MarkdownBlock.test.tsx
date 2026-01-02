import { render, screen } from "@testing-library/react";

import MarkdownBlock from "../../components/MarkdownBlock";

describe("MarkdownBlock", () => {
  it("renders markdown content", () => {
    render(<MarkdownBlock content="**Bold** text" />);
    expect(screen.getByText("Bold")).toBeInTheDocument();
  });
});
