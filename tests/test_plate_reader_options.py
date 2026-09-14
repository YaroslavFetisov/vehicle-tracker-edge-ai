from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("fast_plate_ocr")
pytest.importorskip("open_image_models")

from vehicle_tracker.plate_reader import ALLOW_SPINNING, session_options  # noqa: E402


def test_idle_plate_model_threads_do_not_spin():
    assert session_options().get_session_config_entry(ALLOW_SPINNING) == "0"
