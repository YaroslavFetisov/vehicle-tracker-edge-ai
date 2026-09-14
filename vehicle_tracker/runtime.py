"""Thread and provider settings for the onnx models, and a check for a usable opencv build."""

from __future__ import annotations

# every onnx session defaults to a pool the size of the machine and starves the detector
MAX_PLATE_MODEL_THREADS = 4
CORES_PER_PLATE_THREAD = 4


def plate_model_threads(cores: int | None) -> int:
    return max(1, min(MAX_PLATE_MODEL_THREADS, (cores or 1) // CORES_PER_PLATE_THREAD))


CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def providers_for(device: str, available: list[str]) -> list[str] | None:
    if device == "cuda":
        wanted = [CUDA_PROVIDER, CPU_PROVIDER]
    elif device == "cpu":
        wanted = [CPU_PROVIDER]
    else:
        return None
    # a provider missing from this onnxruntime build makes every session print a warning
    return [provider for provider in wanted if provider in available]


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
