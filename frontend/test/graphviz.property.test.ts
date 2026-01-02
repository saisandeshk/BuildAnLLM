import fc from "fast-check";

import { generateGraphvizArchitecture } from "../lib/graphviz";
import { defaultModelConfig } from "../lib/modelConfig";

describe("graphviz property tests", () => {
  it("always emits the core graph scaffold", () => {
    const configArb = fc.record({
      n_heads: fc.integer({ min: 1, max: 16 }),
      n_kv_heads: fc.integer({ min: 1, max: 16 }),
      positional_encoding: fc.constantFrom("learned", "rope", "alibi", "none"),
      activation: fc.constantFrom("gelu", "swiglu"),
      use_moe: fc.boolean(),
      num_experts: fc.integer({ min: 2, max: 16 }),
      num_experts_per_tok: fc.integer({ min: 1, max: 4 }),
    });

    fc.assert(
      fc.property(configArb, (override) => {
        const cfg = { ...defaultModelConfig, ...override };
        const dot = generateGraphvizArchitecture(cfg);
        expect(dot).toContain("digraph TransformerArchitecture");
        expect(dot).toContain("tokens -> embed");
        expect(dot).toContain("embed -> x0");
      })
    );
  });

  it("annotates RoPE and ALiBi in the attention label", () => {
    const ropeCfg = { ...defaultModelConfig, positional_encoding: "rope" as const };
    const alibiCfg = { ...defaultModelConfig, positional_encoding: "alibi" as const };

    expect(generateGraphvizArchitecture(ropeCfg)).toContain("RoPE");
    expect(generateGraphvizArchitecture(alibiCfg)).toContain("ALiBi");
  });
});
