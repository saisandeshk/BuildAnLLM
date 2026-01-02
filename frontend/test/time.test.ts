import { formatCheckpointTimestamp, formatDuration, formatTimestamp } from "../lib/time";

describe("time helpers", () => {
  it("formats duration across thresholds", () => {
    expect(formatDuration()).toBe("-");
    expect(formatDuration(NaN)).toBe("-");
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(59)).toBe("59s");
    expect(formatDuration(60)).toBe("1m 0s");
    expect(formatDuration(3600)).toBe("1h 0m 0s");
  });

  it("formats timestamps with expected layout", () => {
    const date = new Date(2024, 0, 2, 3, 4, 5);
    expect(formatTimestamp(date)).toBe("240102 03:04:05");
    expect(formatCheckpointTimestamp(date)).toBe("20240102 03:04:05");
  });
});
