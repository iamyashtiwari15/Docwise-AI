import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger once at application startup.
    Format: timestamp | level | module | message
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Avoid adding duplicate handlers if called more than once
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(numeric_level)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "uvicorn.access", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
