import { render, waitFor } from "@testing-library/react";

const initMock = vi.fn();
const useDemoModeMock = vi.fn();

vi.mock("@plausible-analytics/tracker", () => ({
  init: initMock,
}));

vi.mock("../../lib/demo", () => ({
  useDemoMode: useDemoModeMock,
}));

describe("PlausibleAnalytics", () => {
  beforeEach(() => {
    vi.resetModules();
    initMock.mockReset();
    useDemoModeMock.mockReturnValue(false);
  });

  it("does not initialize when demo mode is off", async () => {
    const { default: PlausibleAnalytics } = await import("../../components/PlausibleAnalytics");

    render(<PlausibleAnalytics />);

    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(initMock).not.toHaveBeenCalled();
  });

  it("initializes when demo mode is on", async () => {
    useDemoModeMock.mockReturnValue(true);
    const { default: PlausibleAnalytics } = await import("../../components/PlausibleAnalytics");

    render(<PlausibleAnalytics />);

    await waitFor(() => {
      expect(initMock).toHaveBeenCalledWith({ domain: "buildanllm.com" });
    });
  });
});
