"""How much of the machine the onnx models may use, and which providers they ask for."""

from __future__ import annotations

# Every onnx session opens a thread pool the size of the whole machine by default, so the two
# plate models and the vehicle detector end up fighting over the same cores. Measured on the
# cpu path: 11.5 fps with the defaults against 40.8 fps once the plate models are held to a
# few threads, with detection alone unchanged at 19 ms a frame either way - the detector was
# never slow, it was starved. These models are small and gain nothing from more threads, so
# they take a slice and leave the rest of the machine to the detector.
MAX_PLATE_MODEL_THREADS = 4
CORES_PER_PLATE_THREAD = 4


def plate_model_threads(cores: int | None) -> int:
    return max(1, min(MAX_PLATE_MODEL_THREADS, (cores or 1) // CORES_PER_PLATE_THREAD))


def providers_for(device: str) -> list[str] | None:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cpu":
        return ["CPUExecutionProvider"]
    return None
