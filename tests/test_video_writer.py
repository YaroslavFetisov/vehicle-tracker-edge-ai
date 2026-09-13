from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_tracker.video_writer import VideoWriter

FRAME_COUNT = 5
FRAME_WIDTH = 64
FRAME_HEIGHT = 48


def frame(index: int = 0) -> np.ndarray:
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), index * 20, dtype=np.uint8)


def read_back(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not reopen {path}")
    frames = []
    try:
        while True:
            ok, decoded = capture.read()
            if not ok:
                break
            frames.append(decoded)
    finally:
        capture.release()
    return frames


def test_every_frame_is_written(tmp_path):
    target = tmp_path / "out.mp4"
    with VideoWriter(target, 25.0) as writer:
        for index in range(FRAME_COUNT):
            writer.write(frame(index))

    assert len(read_back(target)) == FRAME_COUNT


def test_the_output_keeps_the_frame_size(tmp_path):
    target = tmp_path / "out.mp4"
    with VideoWriter(target, 25.0) as writer:
        writer.write(frame())

    assert read_back(target)[0].shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)


def test_nothing_is_created_until_a_frame_arrives(tmp_path):
    target = tmp_path / "out.mp4"
    with VideoWriter(target, 25.0):
        pass

    assert not target.exists()


def test_a_missing_directory_is_created(tmp_path):
    target = tmp_path / "results" / "out.mp4"
    with VideoWriter(target, 25.0) as writer:
        writer.write(frame())

    assert target.is_file()


def test_a_source_without_a_frame_rate_still_produces_a_playable_file(tmp_path):
    target = tmp_path / "out.mp4"
    with VideoWriter(target, 0.0) as writer:
        writer.write(frame())

    capture = cv2.VideoCapture(str(target))
    try:
        assert capture.get(cv2.CAP_PROP_FPS) > 0
    finally:
        capture.release()


def test_a_target_that_is_not_a_file_is_refused_before_the_run_starts(tmp_path):
    target = tmp_path / "taken.mp4"
    target.mkdir()

    with pytest.raises(RuntimeError, match="not a file"):
        VideoWriter(target, 25.0)


def test_a_container_the_codec_cannot_fill_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="cannot write video"):
        VideoWriter(tmp_path / "out.txt", 25.0).write(frame())
