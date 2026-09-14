from __future__ import annotations

import pytest

from vehicle_tracker.runtime import (
    MAX_PLATE_MODEL_THREADS,
    gui_is_missing,
    plate_model_threads,
    providers_for,
)

# verbatim from opencv-python-headless 5.0.0.93 and opencv-python 5.0.0.93 on this machine
HEADLESS_BUILD = """
  Parallel framework:            pthreads
  GUI:                           NONE
  Media I/O:
"""
WINDOWED_BUILD = """
  GUI:                           WIN32UI
    Win32 UI:                    YES
  Media I/O:
"""


@pytest.mark.parametrize(("cores", "expected"), [(24, 4), (16, 4), (8, 2), (4, 1), (2, 1), (1, 1)])
def test_plate_models_leave_most_of_the_machine_to_the_detector(cores, expected):
    assert plate_model_threads(cores) == expected


def test_a_machine_that_does_not_report_its_cores_still_gets_a_thread():
    assert plate_model_threads(None) == 1


def test_a_very_large_machine_does_not_hand_the_plate_models_everything():
    assert plate_model_threads(256) == MAX_PLATE_MODEL_THREADS


def test_asking_for_cuda_keeps_the_cpu_provider_as_a_fallback():
    assert providers_for("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cpu_is_pinned_to_the_cpu_provider():
    assert providers_for("cpu") == ["CPUExecutionProvider"]


def test_auto_leaves_the_choice_to_the_runtime():
    assert providers_for("auto") is None


def test_a_headless_build_is_recognised_before_it_fails_on_a_frame():
    assert gui_is_missing(HEADLESS_BUILD)


def test_a_build_with_a_window_backend_is_left_alone():
    assert not gui_is_missing(WINDOWED_BUILD)


def test_a_report_without_a_gui_line_is_given_the_benefit_of_the_doubt():
    assert not gui_is_missing("  Parallel framework:            pthreads\n")


def test_the_gui_line_is_not_confused_with_another_field():
    assert not gui_is_missing("  GUI extras:                    NONE\n")
