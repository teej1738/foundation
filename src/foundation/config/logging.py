"""Structured logging configuration (AD-6).

Configures structlog with timestamper, log level filter, and either
console or JSON rendering. Call configure_logging() once at CLI startup.
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure structlog for the Foundation CLI.

    Parameters
    ----------
    level : str
        Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    json_output : bool
        If True, render as JSON lines (for prod). If False, use colored
        console output (for dev).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Standard library logging config (structlog wraps this)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the formatter for stdlib handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    # Apply formatter to root logger handlers
    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)
