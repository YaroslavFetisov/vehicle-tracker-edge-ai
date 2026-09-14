from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

# the on-screen rate is smoothed, otherwise a single slow frame makes it unreadable
FPS_SMOOTHING = 0.1


@dataclass
class Stage:
    calls: int = 0
    seconds: float = 0.0
    # a stage measured inside another one; counting it again would double the frame budget
    nested: bool = False

    def mean_ms(self) -> float:
        return 0.0 if self.calls == 0 else self.seconds / self.calls * 1000


class Metrics:
    """Wall clock accounting for the pipeline stages.

    Per call and per frame are different questions here: plate reading is expensive per
    call but runs on a fraction of the frames, so only the per frame column says what it
    actually costs the stream.
    """

    def __init__(self, smoothing: float = FPS_SMOOTHING) -> None:
        self._smoothing = smoothing
        self._stages: dict[str, Stage] = {}
        self._frames = 0
        self._frames_without_vehicles = 0
        self._seconds = 0.0
        self._fps = 0.0

    @contextmanager
    def stage(self, name: str, *, nested: bool = False) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - started, nested=nested)

    def record(self, name: str, seconds: float, *, nested: bool = False) -> None:
        stage = self._stages.setdefault(name, Stage(nested=nested))
        stage.calls += 1
        stage.seconds += seconds

    def frame_done(self, seconds: float, *, had_vehicles: bool) -> None:
        self._frames += 1
        self._seconds += seconds
        if not had_vehicles:
            self._frames_without_vehicles += 1
        if seconds > 0:
            instant = 1.0 / seconds
            self._fps = (
                instant
                if self._fps == 0.0
                else (1 - self._smoothing) * self._fps + self._smoothing * instant
            )

    def reset(self) -> None:
        """Drop everything measured so far, used to exclude warm up frames from a benchmark."""
        self._stages.clear()
        self._frames = 0
        self._frames_without_vehicles = 0
        self._seconds = 0.0
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def average_fps(self) -> float:
        return 0.0 if self._seconds <= 0 else self._frames / self._seconds

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def frames_without_vehicles(self) -> int:
        return self._frames_without_vehicles

    def ms_per_frame(self, name: str) -> float:
        stage = self._stages.get(name)
        if stage is None or self._frames == 0:
            return 0.0
        return stage.seconds / self._frames * 1000

    def status_line(self) -> str:
        stages = "  ".join(
            f"{name} {self.ms_per_frame(name):.1f}ms"
            for name, stage in self._stages.items()
            if not stage.nested
        )
        return f"{self._fps:.0f} fps   {stages}"

    def summary(self) -> list[str]:
        # A camera that never opened has nothing to account for, and a table of zeroes
        # printed above the error that ended the run only buries it.
        if self._frames == 0:
            return ["no frames processed"]

        lines = [
            f"{self._frames} frames in {self._seconds:.1f}s, {self.average_fps:.1f} fps average",
            f"frames without vehicles: {self._frames_without_vehicles}",
            f"{'stage':<10}{'calls':>8}{'per frame':>11}{'ms/call':>10}{'ms/frame':>10}",
        ]
        # nested stages last, so that two runs produce tables in the same order
        for name, stage in sorted(self._stages.items(), key=lambda item: item[1].nested):
            lines.append(
                f"{name:<10}{stage.calls:>8}{stage.calls / max(self._frames, 1):>11.2f}"
                f"{stage.mean_ms():>10.1f}{self.ms_per_frame(name):>10.1f}"
            )
        return lines
