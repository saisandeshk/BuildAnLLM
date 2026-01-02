import { render, screen } from "@testing-library/react";

import TrainingControls from "../../components/TrainingControls";

describe("TrainingControls", () => {
  it("shows start label when no job exists", () => {
    render(
      <TrainingControls
        job={null}
        isRunning={false}
        isPaused={false}
        progress={0}
        startLabel="Start Pretrain"
        onPrimary={() => {}}
        onStep={() => {}}
      />
    );

    expect(screen.getByRole("button", { name: "Start Pretrain" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Step" })).toBeDisabled();
  });

  it("shows pause when running and progress when job exists", () => {
    render(
      <TrainingControls
        job={{
          job_id: "job-1",
          kind: "pretrain",
          status: "running",
          iter: 1,
          max_iters: 4,
          created_at: 0,
        }}
        isRunning={true}
        isPaused={false}
        progress={0.25}
        startLabel="Start Pretrain"
        onPrimary={() => {}}
        onStep={() => {}}
      />
    );

    expect(screen.getByRole("button", { name: "Pause" })).toBeEnabled();
    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByText("1 / 4")).toBeInTheDocument();
  });
});
