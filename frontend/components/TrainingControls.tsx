"use client";

import { JobStatus } from "../lib/api";

type TrainingControlsProps = {
  job: JobStatus | null;
  isRunning: boolean;
  isPaused: boolean;
  isCreating?: boolean;
  progress: number;
  startLabel: string;
  error?: string | null;
  onPrimary: () => void;
  onStep: () => void;
};

export default function TrainingControls({
  job,
  isRunning,
  isPaused,
  isCreating = false,
  progress,
  startLabel,
  error,
  onPrimary,
  onStep,
}: TrainingControlsProps) {
  const primaryLabel = isCreating
    ? "Initializing..."
    : !job
    ? startLabel
    : isRunning
    ? "Pause"
    : isPaused
    ? "Resume"
    : "Start New";

  return (
    <div className="card">
      <div className="inline-row" style={{ marginBottom: 12 }}>
        <button className="primary" onClick={onPrimary} disabled={isCreating}>
          {primaryLabel}
        </button>
        <button className="secondary" onClick={onStep} disabled={!job || isRunning}>
          Step
        </button>
      </div>

      {job && (
        <>
          <div className="flex-between" style={{ marginBottom: 8 }}>
            <span className="badge">{job.status}</span>
            <span className="badge">
              {job.iter} / {job.max_iters}
            </span>
          </div>
          <div className="progress">
            <span style={{ width: `${progress * 100}%` }} />
          </div>
        </>
      )}

      {error && <p style={{ color: "#b42318" }}>{error}</p>}
    </div>
  );
}
