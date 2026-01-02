"""Unit tests for tokenizer service helpers."""

import json
from pathlib import Path

import pytest

from backend.app.services.tokenizers import TokenizerLoadError, load_tokenizer_for_checkpoint
from pretraining.tokenization.tokenizer import CharacterTokenizer, SimpleBPETokenizer


@pytest.mark.unit
class TestTokenizerServices:
    def test_load_character_tokenizer(self, tmp_path: Path):
        training_text = tmp_path / "training.txt"
        training_text.write_text("hello world hi", encoding="utf-8")
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_text("ckpt", encoding="utf-8")

        tokenizer = load_tokenizer_for_checkpoint(
            checkpoint_path=checkpoint,
            tokenizer_type="character",
            vocab_size=100,
            training_text_path=training_text,
        )
        assert isinstance(tokenizer, CharacterTokenizer)
        assert tokenizer.decode(tokenizer.encode("hi")) == "hi"

    def test_load_simple_bpe_from_model_file(self, tmp_path: Path):
        training_text = tmp_path / "training.txt"
        training_text.write_text("hello world", encoding="utf-8")
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_text("ckpt", encoding="utf-8")

        model_path = checkpoint.parent / "tokenizer.model"
        data = {
            "vocab": {"a": 0},
            "merges": [],
            "id_to_token": {"0": "a"},
        }
        model_path.write_text(json.dumps(data), encoding="utf-8")

        tokenizer = load_tokenizer_for_checkpoint(
            checkpoint_path=checkpoint,
            tokenizer_type="bpe-simple",
            vocab_size=10,
            training_text_path=training_text,
        )
        assert isinstance(tokenizer, SimpleBPETokenizer)
        assert tokenizer.decode([0]) == "a"

    def test_load_tokenizer_missing_training_text(self, tmp_path: Path):
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_text("ckpt", encoding="utf-8")

        with pytest.raises(TokenizerLoadError):
            load_tokenizer_for_checkpoint(
                checkpoint_path=checkpoint,
                tokenizer_type="character",
                vocab_size=100,
                training_text_path=tmp_path / "missing.txt",
            )

    def test_load_tokenizer_unknown_type(self, tmp_path: Path):
        training_text = tmp_path / "training.txt"
        training_text.write_text("hello world", encoding="utf-8")
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_text("ckpt", encoding="utf-8")

        with pytest.raises(TokenizerLoadError):
            load_tokenizer_for_checkpoint(
                checkpoint_path=checkpoint,
                tokenizer_type="unknown",
                vocab_size=100,
                training_text_path=training_text,
            )
