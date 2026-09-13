from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FOURCC = "mp4v"
# cameras often report no frame rate of their own, and a file still needs one to play back
DEFAULT_FPS = 25.0


class VideoWriter:
    """Writes annotated frames to a file.

    The frame size is taken from the first frame rather than from the source, because a
    stream only reveals its true resolution once a frame has actually arrived.
    """

    def __init__(self, path: Path, fps: float) -> None:
        # checked here rather than on the first frame: opencv reports such a writer as open
        # and then silently writes nothing, which would only surface at the end of a run
        if path.exists() and not path.is_file():
            raise RuntimeError(f"cannot write video to {path}: the path is not a file")
        self._path = path
        if fps <= 0:
            logger.warning("source reports no frame rate, writing at %.0f fps", DEFAULT_FPS)
            fps = DEFAULT_FPS
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            self._writer = self._open(frame.shape[1], frame.shape[0])
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _open(self, width: int, height: int) -> cv2.VideoWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(self._path), cv2.VideoWriter.fourcc(*FOURCC), self._fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot write video to {self._path}")
        logger.info("writing %dx%d at %.0f fps to %s", width, height, self._fps, self._path)
        return writer
