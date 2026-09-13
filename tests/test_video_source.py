from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_tracker.video_source import VideoSource, is_stream

FRAME_COUNT = 12
FRAME_WIDTH = 64
FRAME_HEIGHT = 48


def write_video(path: Path, frame_count: int = FRAME_COUNT) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25.0, (FRAME_WIDTH, FRAME_HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError("could not create the test video")
    for index in range(frame_count):
        writer.write(np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), index * 20, dtype=np.uint8))
    writer.release()
    return path


def test_file_source_yields_every_frame(tmp_path):
    video = write_video(tmp_path / "clip.mp4")
    with VideoSource(str(video)) as source:
        frames = list(source)
    assert len(frames) == FRAME_COUNT


def test_frames_keep_their_shape(tmp_path):
    video = write_video(tmp_path / "clip.mp4")
    with VideoSource(str(video)) as source:
        frame = next(iter(source))
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)


def test_file_source_preserves_frame_order(tmp_path):
    video = write_video(tmp_path / "clip.mp4")
    with VideoSource(str(video)) as source:
        brightness = [float(frame.mean()) for frame in source]
    assert brightness == sorted(brightness)


def test_unopenable_source_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="cannot open"):
        VideoSource(str(tmp_path / "missing.mp4")).start()


def test_iterating_before_start_is_reported(tmp_path):
    video = write_video(tmp_path / "clip.mp4")
    with pytest.raises(RuntimeError, match="not started"):
        next(iter(VideoSource(str(video))))


@pytest.mark.parametrize("source", ["rtsp://camera/stream", "http://camera/stream", 0])
def test_live_sources_are_recognised(source):
    assert is_stream(source)


def test_file_path_is_not_a_live_source():
    assert not is_stream("data/highway_traffic.mp4")
