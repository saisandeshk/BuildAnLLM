import { finetuneEquations, inferenceEquations, loraEquations, modelEquations } from "../lib/equations";

describe("equations content", () => {
  it("includes key model sections", () => {
    expect(modelEquations).toContain("Token embedding");
    expect(modelEquations).toContain("Attention");
    expect(modelEquations).toContain("MLP");
  });

  it("includes inference and fine-tune notes", () => {
    expect(inferenceEquations).toContain("Temperature");
    expect(finetuneEquations).toContain("Masked loss");
    expect(loraEquations).toContain("LoRA adaptation");
  });
});
