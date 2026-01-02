"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export default function Heatmap({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  const size = matrix.length;
  const axisLabels = labels.slice(0, size);
  const labelWidth = 120;
  const topLabelHeight = 64;
  const availableWidth = Math.max(0, (width || 0) - labelWidth);
  const cell = Math.max(2, Math.min(14, Math.floor(availableWidth / (size || 1)) || 10));
  const canvasSize = size * cell;

  const topGridStyle = useMemo(
    () => ({
      gridTemplateColumns: `repeat(${size}, ${cell}px)`,
      width: `${canvasSize}px`,
      height: `${topLabelHeight}px`,
    }),
    [cell, canvasSize, size]
  );

  const leftGridStyle = useMemo(
    () => ({
      gridTemplateRows: `repeat(${size}, ${cell}px)`,
      width: `${labelWidth}px`,
      height: `${canvasSize}px`,
    }),
    [cell, canvasSize]
  );

  const gridStyle = useMemo(
    () => ({
      gridTemplateColumns: `${labelWidth}px ${canvasSize}px`,
      gridTemplateRows: `${topLabelHeight}px ${canvasSize}px`,
    }),
    [canvasSize]
  );

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) {
      return;
    }

    const updateWidth = () => {
      setWidth(wrapper.getBoundingClientRect().width);
    };

    updateWidth();

    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateWidth);
      return () => {
        window.removeEventListener("resize", updateWidth);
      };
    }

    const observer = new ResizeObserver((entries) => {
      entries.forEach((entry) => {
        setWidth(entry.contentRect.width);
      });
    });
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || matrix.length === 0) {
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    canvas.width = canvasSize * ratio;
    canvas.height = canvasSize * ratio;
    canvas.style.width = `${canvasSize}px`;
    canvas.style.height = `${canvasSize}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

    const max = Math.max(...matrix.flat());
    const min = Math.min(...matrix.flat());
    const range = max - min || 1;

    for (let i = 0; i < size; i += 1) {
      for (let j = 0; j < size; j += 1) {
        const value = (matrix[i][j] - min) / range;
        const hue = 18;
        const light = 90 - value * 55;
        ctx.fillStyle = `hsl(${hue}deg 70% ${light}%)`;
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    for (let i = 0; i <= size; i += 1) {
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(size * cell, i * cell);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(i * cell, 0);
      ctx.lineTo(i * cell, size * cell);
      ctx.stroke();
    }
  }, [matrix, canvasSize]);

  return (
    <div ref={wrapperRef} className="heatmap-wrapper">
      <div className="heatmap-grid" style={gridStyle}>
        <div className="heatmap-corner">
          <span className="heatmap-axis-label">Query ↓</span>
          <span className="heatmap-axis-label">Key →</span>
        </div>
        <div className="heatmap-top" style={topGridStyle}>
          {axisLabels.map((label, idx) => (
            <span key={`x-${label}-${idx}`} className="heatmap-token heatmap-token-x">
              {label}
            </span>
          ))}
        </div>
        <div className="heatmap-left" style={leftGridStyle}>
          {axisLabels.map((label, idx) => (
            <span key={`y-${label}-${idx}`} className="heatmap-token heatmap-token-y">
              {label}
            </span>
          ))}
        </div>
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
