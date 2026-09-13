from __future__ import annotations

import logging

from vehicle_tracker.config import parse_args
from vehicle_tracker.pipeline import run


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
