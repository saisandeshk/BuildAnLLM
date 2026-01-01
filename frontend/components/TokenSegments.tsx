"use client";

export default function TokenSegments({
  tokens,
  tone,
}: {
  tokens: string[];
  tone: "prompt" | "response" | "neutral";
}) {
  const palette = {
    prompt: "rgba(20, 33, 43, 0.12)",
    response: "rgba(78, 205, 196, 0.35)",
    neutral: "rgba(210, 75, 26, 0.2)",
  } as const;

  return (
    <div style={{ lineHeight: 1.8, fontFamily: "JetBrains Mono, Courier New, monospace" }}>
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
