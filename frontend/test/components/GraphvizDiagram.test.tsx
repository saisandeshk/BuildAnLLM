import { render, screen, waitFor } from "@testing-library/react";

import GraphvizDiagram from "../../components/GraphvizDiagram";

const renderStringMock = vi.fn();

vi.mock("@viz-js/viz", () => ({
  instance: vi.fn(async () => ({
    renderString: renderStringMock,
  })),
}));

describe("GraphvizDiagram", () => {
  beforeEach(() => {
    renderStringMock.mockReset();
  });

  it("renders SVG output", async () => {
    renderStringMock.mockReturnValue("<svg>ok</svg>");
    const { container } = render(<GraphvizDiagram dot="digraph{}" />);

    await waitFor(() => {
      expect(container.querySelector("svg")).toBeTruthy();
    });
  });

  it("renders errors from the renderer", async () => {
    renderStringMock.mockImplementation(() => {
      throw new Error("render failed");
    });
    render(<GraphvizDiagram dot="digraph{}" />);

    expect(await screen.findByText("render failed")).toBeInTheDocument();
  });
});
