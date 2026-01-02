"use client";

type LogBoxProps = {
  logs: string[];
  emptyText?: string;
};

export default function LogBox({ logs, emptyText = "No logs yet." }: LogBoxProps) {
  return <div className="log-box">{logs.length === 0 ? emptyText : logs.join("\n")}</div>;
}
