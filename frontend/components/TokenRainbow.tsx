"use client";

import { useMemo } from "react";

function colorForIndex(idx: number) {
  const hue = (idx * 47) % 360;
  return `hsl(${hue}deg 70% 80%)`;
}

export default function TokenRainbow({ tokens }: { tokens: string[] }) {
  const colors = useMemo(() => tokens.map((_, idx) => colorForIndex(idx)), [tokens]);
  return (
    <div style={{ lineHeight: 1.8, fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace" }}>
      {tokens.map((token, idx) => (
        <span
          key={`${token}-${idx}`}
          style={{
            background: colors[idx],
            padding: "2px 6px",
            marginRight: 4,
            borderRadius: 6,
            display: "inline-block",
            color: "#0b0b0c",
          }}
        >
          {token || " "}
        </span>
      ))}
    </div>
  );
}
