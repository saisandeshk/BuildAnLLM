import {
  applyPreset,
  applySizePreset,
  defaultModelConfig,
  estimateParams,
} from "../lib/modelConfig";

describe("modelConfig helpers", () => {
  it("applies GPT preset without MoE", () => {
    const cfg = applyPreset({ ...defaultModelConfig, use_moe: true }, "gpt");
    expect(cfg.positional_encoding).toBe("learned");
    expect(cfg.normalization).toBe("layernorm");
    expect(cfg.activation).toBe("gelu");
    expect(cfg.use_moe).toBe(false);
  });

  it("applies LLaMA preset with MoE + GQA", () => {
    const cfg = applyPreset({ ...defaultModelConfig, n_heads: 8 }, "llama");
    expect(cfg.positional_encoding).toBe("rope");
    expect(cfg.normalization).toBe("rmsnorm");
    expect(cfg.activation).toBe("swiglu");
    expect(cfg.use_moe).toBe(true);
    expect(cfg.n_kv_heads).toBe(Math.max(1, Math.floor(8 / 4)));
  });

  it("applies size presets and estimates params", () => {
    const cfg = applySizePreset(defaultModelConfig, "small");
    expect(cfg.d_model).toBe(256);
    expect(cfg.n_kv_heads).toBe(cfg.n_heads);
    expect(estimateParams(cfg)).toBeGreaterThan(0);
  });
});
