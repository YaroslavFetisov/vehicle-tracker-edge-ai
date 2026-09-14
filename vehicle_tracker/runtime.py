"""How much of the machine the onnx models may use, which providers they ask for, and
whether the opencv build that came with them can open a window at all."""

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


# Both plate model packages depend on opencv-python-headless, which ships the same cv2 module
# as opencv-python and overwrites it, so which of the two a fresh install ends up with is
# decided by the order pip happens to resolve them in. The headless build has no window
# support compiled in and raises "the function is not implemented" on the first frame, which
# is a poor way to learn this when displaying is the default mode. A build names its window
# backend on the GUI line of its build information - WIN32UI, GTK2, COCOA, QT - and says NONE
# when it has none. Only that outright NONE counts as missing here: a report this does not
# recognise is left alone and fails later with opencv's own message, if it fails at all.
NO_GUI = "NONE"
GUI_FIELD = "GUI"


def gui_is_missing(build_information: str) -> bool:
    """Whether this OpenCV build states outright that it cannot open a window."""
    for line in build_information.splitlines():
        field, separator, backend = line.partition(":")
        if separator and field.strip() == GUI_FIELD:
            return backend.strip().upper() == NO_GUI
    return False
