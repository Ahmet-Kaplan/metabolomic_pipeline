"""structlog + stdlib logging bootstrap.

Call :func:`configure_logging` once per process, at CLI entry. After that, every
module can use ``structlog.get_logger(__name__)`` and produce structured logs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    include_caller: bool = True,
    log_file: Path | None = None,
) -> None:
    """Configure structlog + stdlib logging for the whole process.

    Parameters
    ----------
    level
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    json_output
        If True, render log records as one JSON object per line (machine-
        friendly). If False, use a coloured key-value console renderer
        (dev-friendly).
    include_caller
        Whether to attach filename/lineno to each record.
    log_file
        Optional path; if given, JSON logs are duplicated to this file.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    # Note: we deliberately omit ``add_logger_name`` because PrintLogger (our
    # factory) has no ``.name`` attribute. If/when we switch to stdlib logger
    # factory for file handlers, re-add that processor.
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]
    if include_caller:
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
        )

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Silence stdlib noise (third-party libs that still use logging)
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)
