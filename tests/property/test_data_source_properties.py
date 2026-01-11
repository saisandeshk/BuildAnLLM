"""Property-based tests for data source selection functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest
from hypothesis import given, strategies as st, settings


@pytest.mark.property
class TestDataSourceProperties:
    """Property-based tests for data source handling."""

    @given(
        num_files=st.integers(min_value=1, max_value=3),
        content_length=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=10)
    def test_read_training_text_combines_nonempty_results(
        self, num_files: int, content_length: int
    ):
        """Property: combining any number of non-empty files produces non-empty result."""
        from backend.app.routers import pretrain

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = []
            for i in range(num_files):
                file_path = tmp_path / f"file_{i}.txt"
                file_path.write_text("x" * content_length, encoding="utf-8")
                paths.append(str(file_path))

            result = pretrain._read_training_text(None, paths)

            # Result should not be empty if we had valid paths
            assert len(result) > 0
            # Result should contain content from all files
            assert result.count("x") >= content_length

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=3))
    @settings(max_examples=20)
    def test_training_text_paths_field_accepts_any_string_list(self, paths: List[str]):
        """Property: training_text_paths accepts any list of strings."""
        from backend.app.schemas.training import PretrainJobPayload

        payload = PretrainJobPayload(
            model_config={
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 2,
                "n_ctx": 64,
                "d_head": 32,
                "d_mlp": 128,
                "vocab_size": 100,
            },
            training_text_paths=paths,
        )

        assert payload.training_text_paths == paths


@pytest.mark.property
class TestDataSourceMetadataProperties:
    """Property tests for data source metadata consistency."""

    def test_all_sources_have_consistent_structure(self):
        """All data sources must have the same set of fields."""
        from backend.app.routers.pretrain import PRETRAINING_DATA_SOURCES

        if not PRETRAINING_DATA_SOURCES:
            pytest.skip("No data sources configured")

        # Get fields from first source
        first_source = next(iter(PRETRAINING_DATA_SOURCES.values()))
        expected_fields = set(first_source.keys())

        # All sources should have the same fields
        for name, info in PRETRAINING_DATA_SOURCES.items():
            actual_fields = set(info.keys())
            assert actual_fields == expected_fields, (
                f"Source '{name}' has inconsistent fields. "
                f"Expected {expected_fields}, got {actual_fields}"
            )

    def test_all_filenames_are_valid_paths(self):
        """All filename fields should be valid path strings."""
        from backend.app.routers.pretrain import PRETRAINING_DATA_SOURCES

        for name, info in PRETRAINING_DATA_SOURCES.items():
            filename = info["filename"]
            # Should be a non-empty string
            assert isinstance(filename, str), f"'{name}' filename is not a string"
            assert len(filename) > 0, f"'{name}' has empty filename"
            # Should look like a path
            assert "/" in filename or filename.endswith(".txt"), (
                f"'{name}' filename doesn't look like a path: {filename}"
            )

    def test_languages_are_non_empty_strings(self):
        """All language fields should be non-empty strings."""
        from backend.app.routers.pretrain import PRETRAINING_DATA_SOURCES

        for name, info in PRETRAINING_DATA_SOURCES.items():
            language = info["language"]
            assert isinstance(language, str), f"'{name}' language is not a string"
            assert len(language) > 0, f"'{name}' has empty language"

    def test_scripts_are_non_empty_strings(self):
        """All script fields should be non-empty strings."""
        from backend.app.routers.pretrain import PRETRAINING_DATA_SOURCES

        for name, info in PRETRAINING_DATA_SOURCES.items():
            script = info["script"]
            assert isinstance(script, str), f"'{name}' script is not a string"
            assert len(script) > 0, f"'{name}' has empty script"
