from __future__ import annotations

import cv2
import numpy as np

from vehicle_tracker.detection import Detection

BOX_COLOR = (0, 220, 0)
BOX_THICKNESS = 2
LABEL_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1
LABEL_PADDING = 4


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> None:
    """Annotates the frame in place."""
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        draw_label(frame, f"ID {detection.track_id}", (x1, y1), BOX_COLOR)


def draw_label(
    frame: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    background: tuple[int, int, int],
) -> None:
    x, y = anchor
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    box_height = text_height + 2 * LABEL_PADDING
    top = max(y - box_height, 0)
    cv2.rectangle(
        frame,
        (x, top),
        (x + text_width + 2 * LABEL_PADDING, top + box_height),
        background,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        text,
        (x + LABEL_PADDING, top + text_height + LABEL_PADDING - 1),
        FONT,
        FONT_SCALE,
        LABEL_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
