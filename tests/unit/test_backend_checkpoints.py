"""Unit tests for checkpoint service helpers."""

from pathlib import Path

import pytest

from backend.app.services.checkpoints import list_checkpoints, resolve_checkpoint_path, _extract_iter


@pytest.mark.unit
class TestCheckpointServices:
    def test_extract_iter(self):
        assert _extract_iter("checkpoint_42.pt") == 42
        assert _extract_iter("checkpoint_notanumber.pt") is None
        assert _extract_iter("final_model.pt") is None

    def test_list_checkpoints_includes_pretrain_and_sft(self, tmp_path: Path):
        base = tmp_path / "checkpoints"
        run_a = base / "20240101"
        run_a.mkdir(parents=True)
        (run_a / "checkpoint_2.pt").write_text("a", encoding="utf-8")
        (run_a / "checkpoint_1.pt").write_text("b", encoding="utf-8")
        run_a_sft = run_a / "sft"
        run_a_sft.mkdir()
        (run_a_sft / "checkpoint_3.pt").write_text("c", encoding="utf-8")

        run_b = base / "20240102"
        run_b.mkdir()
        (run_b / "final_model.pt").write_text("d", encoding="utf-8")
        run_b_sft = run_b / "sft"
        run_b_sft.mkdir()
        (run_b_sft / "final_model.pt").write_text("e", encoding="utf-8")

        results = list_checkpoints(str(base))
        paths = {entry["path"] for entry in results}
        assert str(run_a / "checkpoint_2.pt") in paths
        assert str(run_a / "checkpoint_1.pt") in paths
        assert str(run_a_sft / "checkpoint_3.pt") in paths
        assert str(run_b / "final_model.pt") in paths
        assert str(run_b_sft / "final_model.pt") in paths

        finetuned = {entry["path"]: entry["is_finetuned"] for entry in results}
        assert finetuned[str(run_a_sft / "checkpoint_3.pt")] is True
        assert finetuned[str(run_b_sft / "final_model.pt")] is True
        assert finetuned[str(run_a / "checkpoint_1.pt")] is False

    def test_resolve_checkpoint_path_scoped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        base = tmp_path / "checkpoints"
        base.mkdir()
        inside = base / "model.pt"
        inside.write_text("ok", encoding="utf-8")
        assert resolve_checkpoint_path(str(inside)) == inside.resolve()

        with pytest.raises(FileNotFoundError):
            resolve_checkpoint_path(str(base / "missing.pt"))

        outside = tmp_path / "outside.pt"
        outside.write_text("nope", encoding="utf-8")
        with pytest.raises(ValueError):
            resolve_checkpoint_path(str(outside))
