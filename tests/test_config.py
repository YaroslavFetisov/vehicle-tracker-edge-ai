from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from vehicle_tracker.config import parse_args, parse_source


def test_rtsp_url_is_kept_as_is():
    url = "rtsp://10.0.0.5:554/stream"
    assert parse_source(url) == url


def test_digits_become_webcam_index():
    assert parse_source("0") == 0


def test_existing_file_is_accepted(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    assert parse_source(str(video)) == str(video)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_source(str(tmp_path / "nope.mp4"))


def test_defaults_to_auto_device():
    config = parse_args(["--source", "rtsp://cam/stream"])
    assert config.device == "auto"


def test_device_can_be_forced():
    config = parse_args(["--source", "rtsp://cam/stream", "--device", "cpu"])
    assert config.device == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--source", "rtsp://cam/stream", "--device", "tpu"])


def test_the_window_is_shown_by_default():
    config = parse_args(["--source", "rtsp://cam/stream"])
    assert config.display
    assert config.output is None


def test_display_can_be_turned_off():
    config = parse_args(["--source", "rtsp://cam/stream", "--no-display"])
    assert not config.display


def test_output_file_is_optional_and_kept_as_a_path():
    config = parse_args(["--source", "rtsp://cam/stream", "--save-video", "out/run.mp4"])
    assert config.output == Path("out/run.mp4")
