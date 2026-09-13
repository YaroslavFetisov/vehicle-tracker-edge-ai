from __future__ import annotations

import logging

from vehicle_tracker.config import parse_args
from vehicle_tracker.pipeline import run

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run(parse_args())
    except RuntimeError as exc:
        # losing a camera for good is an operating condition, not a defect in the program:
        # it deserves one clear line and a non zero exit code, not a stack trace
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
