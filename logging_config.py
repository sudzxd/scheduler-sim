"""Logging configuration for scheduler simulator.

This module provides centralised logging setup with consistent formatting
across all modules.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

# Standard library
import logging
import sys

# =============================================================================
# 2. CONSTANTS
# =============================================================================

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# 3. PUBLIC API
# =============================================================================


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the entire application.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
