import { API_BASE_URL } from "../lib/env";
import { fetchJson, makeFormData } from "../lib/api";

describe("api helpers", () => {
  it("fetchJson returns json for ok responses", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ ok: true }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const result = await fetchJson<{ ok: boolean }>("/status");
    expect(result).toEqual({ ok: true });
    expect(mockFetch).toHaveBeenCalledWith(`${API_BASE_URL}/status`, undefined);
  });

  it("fetchJson throws for non-ok responses", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => "boom",
    });
    vi.stubGlobal("fetch", mockFetch);

    await expect(fetchJson("/status")).rejects.toThrow("boom");
  });

  it("makeFormData stores payload and file", () => {
    const payload = { a: 1 };
    const file = new File(["hello"], "hello.txt", { type: "text/plain" });
    const form = makeFormData(payload, file, "data");

    expect(form.get("payload")).toBe(JSON.stringify(payload));
    expect(form.get("data")).toBe(file);
  });
});
