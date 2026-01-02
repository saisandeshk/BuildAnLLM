import { render, screen } from "@testing-library/react";

import CodePanel from "../../components/CodePanel";
import type { CodeSnippet } from "../../lib/api";

describe("CodePanel", () => {
  it("renders snippet metadata and code", () => {
    const snippet: CodeSnippet = {
      title: "1. Model",
      module: "pretraining.model.model",
      object: "TransformerModel",
      file: "pretraining/model.py",
      start_line: 10,
      end_line: 20,
      github_url: "https://example.test/model",
      code: "class TransformerModel:\n    pass",
    };

    render(<CodePanel snippet={snippet} />);

    expect(screen.getByText("1. Model")).toBeInTheDocument();
    expect(screen.getByText("view source")).toHaveAttribute("href", snippet.github_url);
    expect(screen.getByText("pretraining/model.py:10-20")).toBeInTheDocument();
    expect(screen.getByText(/class TransformerModel:/)).toBeInTheDocument();
  });
});
