"""Test-wide safety: no test may read or write the user's real calibration."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_calibration_store(tmp_path, monkeypatch):
    # CalibrationStore() without an explicit path resolves default_store_path() at
    # construction time; point it at a temporary file for every test.
    monkeypatch.setattr(
        "handpiano.mapping.calibration.default_store_path", lambda: tmp_path / "isolated-calibration.json"
    )
