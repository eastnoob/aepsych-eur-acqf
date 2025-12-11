from __future__ import annotations
from typing import Optional
import sys
import io
from loguru import logger


def configure_logger(level: str = "WARNING", log_file: Optional[str] = None) -> None:
    """Configure loguru global logger for this extension.

    Args:
        level: Minimum log level as string (DEBUG/INFO/WARNING/ERROR)
        log_file: Optional path to a file to also write logs to.
    """
    # Remove default handlers and reconfigure
    logger.remove()

    # On Windows with GBK encoding, wrap stdout with UTF-8 to handle Chinese characters
    output_stream = sys.stdout
    if sys.platform == "win32" and hasattr(sys.stdout, "encoding"):
        # Check if stdout is using GBK or similar non-UTF8 encoding
        stdout_encoding = sys.stdout.encoding or ""
        if "utf" not in stdout_encoding.lower():
            # Wrap stdout with UTF-8, using 'replace' to handle unencodable characters
            output_stream = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",  # Replace unencodable chars with '?'
                line_buffering=True,
            )

    # Console sink using the configured output stream
    logger.add(output_stream, level=level, colorize=False)

    # Optional file sink (rotation/retention can be added by caller)
    if log_file:
        logger.add(log_file, level=level, rotation="10 MB", retention="7 days")


__all__ = ["configure_logger"]
