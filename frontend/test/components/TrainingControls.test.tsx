import { render, screen, fireEvent } from "@testing-library/react";
import { vi } from "vitest";

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

  describe("Stop/Reset button", () => {
    it("shows Stop/Reset button when onReset prop is provided", () => {
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
          onReset={() => {}}
        />
      );

      expect(screen.getByRole("button", { name: "Stop / Reset" })).toBeInTheDocument();
    });

    it("does not show Stop/Reset button when onReset prop is not provided", () => {
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

      expect(screen.queryByRole("button", { name: "Stop / Reset" })).not.toBeInTheDocument();
    });

    it("disables Stop/Reset button when no job exists", () => {
      render(
        <TrainingControls
          job={null}
          isRunning={false}
          isPaused={false}
          progress={0}
          startLabel="Start Pretrain"
          onPrimary={() => {}}
          onStep={() => {}}
          onReset={() => {}}
        />
      );

      expect(screen.getByRole("button", { name: "Stop / Reset" })).toBeDisabled();
    });

    it("enables Stop/Reset button when job exists", () => {
      render(
        <TrainingControls
          job={{
            job_id: "job-1",
            kind: "pretrain",
            status: "paused",
            iter: 10,
            max_iters: 100,
            created_at: 0,
          }}
          isRunning={false}
          isPaused={true}
          progress={0.1}
          startLabel="Start Pretrain"
          onPrimary={() => {}}
          onStep={() => {}}
          onReset={() => {}}
        />
      );

      expect(screen.getByRole("button", { name: "Stop / Reset" })).toBeEnabled();
    });

    it("calls onReset when Stop/Reset button is clicked", () => {
      const onResetMock = vi.fn();
      render(
        <TrainingControls
          job={{
            job_id: "job-1",
            kind: "pretrain",
            status: "paused",
            iter: 10,
            max_iters: 100,
            created_at: 0,
          }}
          isRunning={false}
          isPaused={true}
          progress={0.1}
          startLabel="Start Pretrain"
          onPrimary={() => {}}
          onStep={() => {}}
          onReset={onResetMock}
        />
      );

      fireEvent.click(screen.getByRole("button", { name: "Stop / Reset" }));
      expect(onResetMock).toHaveBeenCalledTimes(1);
    });
  });
});
