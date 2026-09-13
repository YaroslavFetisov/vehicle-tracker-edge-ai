from __future__ import annotations

import cv2
import numpy as np

from vehicle_tracker.detection import Detection
from vehicle_tracker.tracks import TrackState

UNKNOWN_COLOR = (0, 220, 0)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
LABEL_PADDING = 4
DARK_TEXT = (0, 0, 0)
LIGHT_TEXT = (255, 255, 255)
LUMINANCE_THRESHOLD = 140


def text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = background
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return DARK_TEXT if luminance > LUMINANCE_THRESHOLD else LIGHT_TEXT


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    states: dict[int, TrackState],
) -> None:
    """Annotates the frame in place."""
    # labels are wider than a distant vehicle's box and would overwrite each other in
    # traffic, so the nearest vehicles are drawn last and stay readable
    for detection in sorted(detections, key=box_area):
        state = states.get(detection.track_id)
        color = state.color if state is not None else None
        box_color = color.bgr if color is not None else UNKNOWN_COLOR

        label = f"ID {detection.track_id}"
        if color is not None:
            label = f"{label}  {color.name}"
        if state is not None and state.plate is not None:
            label = f"{label}  {state.plate}"

        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, BOX_THICKNESS)
        draw_label(frame, label, (x1, y1), box_color)


def box_area(detection: Detection) -> int:
    x1, y1, x2, y2 = detection.bbox
    return (x2 - x1) * (y2 - y1)


def draw_label(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    background: tuple[int, int, int],
) -> None:
    x, y = anchor
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    box_width = text_width + 2 * LABEL_PADDING
    box_height = text_height + 2 * LABEL_PADDING
    # a label is wider than the box of a vehicle at the edge of the frame, and the plate
    # is at the end of it, so the label slides inwards rather than being clipped away
    x = max(min(x, frame.shape[1] - box_width), 0)
    top = max(y - box_height, 0)
    cv2.rectangle(frame, (x, top), (x + box_width, top + box_height), background, cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (x + LABEL_PADDING, top + text_height + LABEL_PADDING - 1),
        FONT,
        FONT_SCALE,
        text_color(background),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
