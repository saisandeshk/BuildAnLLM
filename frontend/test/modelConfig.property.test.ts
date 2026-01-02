import fc from "fast-check";

import { applySizePreset, defaultModelConfig } from "../lib/modelConfig";

describe("modelConfig property tests", () => {
  it("applySizePreset always aligns n_kv_heads with n_heads", () => {
    const sizeArb = fc.constantFrom("small", "medium", "full");
    const overrides = fc.record({
      d_model: fc.integer({ min: 8, max: 2048 }),
      n_heads: fc.integer({ min: 1, max: 16 }),
      n_layers: fc.integer({ min: 1, max: 12 }),
      n_ctx: fc.integer({ min: 8, max: 2048 }),
      d_head: fc.integer({ min: 8, max: 256 }),
      d_mlp: fc.integer({ min: 8, max: 4096 }),
    });

    fc.assert(
      fc.property(sizeArb, overrides, (size, override) => {
        const base = { ...defaultModelConfig, ...override };
        const sized = applySizePreset(base, size);
        expect(sized.n_kv_heads).toBe(sized.n_heads);
      })
    );
  });
});
