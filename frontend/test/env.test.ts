describe("env", () => {
  const original = process.env.NEXT_PUBLIC_API_BASE_URL;

  afterEach(() => {
    if (original === undefined) {
      delete process.env.NEXT_PUBLIC_API_BASE_URL;
    } else {
      process.env.NEXT_PUBLIC_API_BASE_URL = original;
    }
  });

  it("uses default API base when env is unset", async () => {
    delete process.env.NEXT_PUBLIC_API_BASE_URL;
    vi.resetModules();
    const { API_BASE_URL } = await import("../lib/env");
    expect(API_BASE_URL).toBe("http://127.0.0.1:8000");
  });

  it("uses env override when provided", async () => {
    process.env.NEXT_PUBLIC_API_BASE_URL = "http://example.test";
    vi.resetModules();
    const { API_BASE_URL } = await import("../lib/env");
    expect(API_BASE_URL).toBe("http://example.test");
  });
});
