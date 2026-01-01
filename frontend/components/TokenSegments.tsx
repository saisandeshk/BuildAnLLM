"use client";

export default function TokenSegments({
  tokens,
  tone,
}: {
  tokens: string[];
  tone: "prompt" | "response" | "neutral";
}) {
  const palette = {
    prompt: "rgba(255, 255, 255, 0.08)",
    response: "rgba(249, 115, 22, 0.3)",
    neutral: "rgba(249, 115, 22, 0.18)",
  } as const;

  return (
    <div style={{ lineHeight: 1.8, fontFamily: "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace" }}>
      {tokens.map((token, idx) => (
        <span
          key={`${token}-${idx}`}
          style={{
            background: palette[tone],
            padding: "2px 6px",
            marginRight: 4,
            borderRadius: 6,
            display: "inline-block",
          }}
        >
          {token || " "}
        </span>
      ))}
    </div>
  );
}
