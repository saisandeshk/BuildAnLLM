import { render, screen } from "@testing-library/react";

import ModelConfigSummary from "../../components/ModelConfigSummary";

describe("ModelConfigSummary", () => {
  it("renders derived attention type and relevant sections", () => {
    const config = {
      positional_encoding: "rope",
      normalization: "rmsnorm",
      activation: "swiglu",
      n_heads: 4,
      n_kv_heads: 1,
      n_layers: 2,
      d_model: 256,
      d_head: 64,
      d_mlp: 1024,
      n_ctx: 128,
      use_moe: true,
      num_experts: 8,
      num_experts_per_tok: 2,
      router_type: "top_k",
      rope_theta: 10000,
    };

    render(<ModelConfigSummary config={config} />);

    expect(screen.getByText("Attention Type")).toBeInTheDocument();
    expect(screen.getByText("Multi-Query (MQA)")).toBeInTheDocument();
    expect(screen.getByText("RoPE theta")).toBeInTheDocument();
    expect(screen.getByText("Experts")).toBeInTheDocument();
  });
});
