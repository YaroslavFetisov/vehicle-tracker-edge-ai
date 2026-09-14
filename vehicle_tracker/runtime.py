"""Thread and provider settings for the onnx models, and a check for a usable opencv build."""

from __future__ import annotations

# every onnx session defaults to a pool the size of the machine and starves the detector
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


# The plate model packages pull in opencv-python-headless, which can replace opencv-python
# and has no window support. Only an explicit NONE is treated as missing.
NO_GUI = "NONE"
GUI_FIELD = "GUI"


def gui_is_missing(build_information: str) -> bool:
    """Whether this OpenCV build states outright that it cannot open a window."""
    for line in build_information.splitlines():
        field, separator, backend = line.partition(":")
        if separator and field.strip() == GUI_FIELD:
            return backend.strip().upper() == NO_GUI
    return False
