"""Unit tests for system info helpers."""

import pytest

from backend.app.services.system_info import get_system_info


@pytest.mark.unit
def test_get_system_info_keys():
    info = get_system_info()
    expected = {"cpu", "ram_gb", "gpu", "mps", "os", "python", "torch"}
    assert expected.issubset(info.keys())
    for key in expected:
        assert isinstance(info[key], str)
