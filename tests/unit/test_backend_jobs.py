"""Unit tests for job and registry helpers."""

import time

import pytest

from backend.app.core.jobs import InferenceRegistry, JobRegistry, TrainingJob


class DummyTrainer:
    def __init__(self, max_iters: int = 2) -> None:
        self.max_iters = max_iters
        self.calls = 0
        self.saved = []
        self.saved_graph = False

    def train_single_step(self):
        self.calls += 1
        return {
            "loss": 1.0,
            "running_loss": 1.0,
            "grad_norm": 0.1,
        }

    def save_checkpoint(self, step: int, is_final: bool = False):
        self.saved.append((step, is_final))

    def save_loss_graph(self):
        self.saved_graph = True


@pytest.mark.unit
class TestTrainingJob:
    def test_step_once_completes_and_saves(self):
        trainer = DummyTrainer(max_iters=2)
        job = TrainingJob(
            job_id="job-1",
            kind="pretrain",
            trainer=trainer,
            eval_interval=0,
            save_interval=0,
        )

        payload = job.step_once()
        assert payload["iter"] == 1
        assert job.status == "paused"

        payload = job.step_once()
        assert payload["iter"] == 2
        assert job.status == "completed"
        assert trainer.saved[-1] == (2, True)
        assert trainer.saved_graph is True

    def test_pause_and_resume_update_status(self):
        trainer = DummyTrainer(max_iters=1)
        job = TrainingJob(
            job_id="job-2",
            kind="pretrain",
            trainer=trainer,
            eval_interval=0,
            save_interval=0,
        )
        job.status = "running"
        job.started_at = time.time()
        job.pause()
        assert job.status == "paused"
        job.resume()
        assert job.status == "running"


@pytest.mark.unit
def test_job_registry_crud():
    registry = JobRegistry()
    trainer = DummyTrainer()
    job = TrainingJob(
        job_id="job-3",
        kind="pretrain",
        trainer=trainer,
        eval_interval=0,
        save_interval=0,
    )
    registry.add(job)
    assert registry.get("job-3") is job
    registry.remove("job-3")
    assert registry.get("job-3") is None


@pytest.mark.unit
def test_inference_registry_crud():
    registry = InferenceRegistry()
    class DummySession:
        def __init__(self, session_id: str) -> None:
            self.session_id = session_id

    session = DummySession("session-1")
    registry.add(session)
    assert registry.get("session-1") is session
    registry.remove("session-1")
    assert registry.list() == {}
