import logging
import sys
from logging import LogRecord
from pathlib import Path


class CustomFormatter(logging.Formatter):
    """Format log records with compact visual severity markers.

    Attributes
    ----------
    ICONS : dict[str, str]
        Mapping from logging level names to prefixes used in formatted messages.
    """

    ICONS = {
        "DEBUG": "🔍",
        "INFO": "✨",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "💀",
    }

    def format(self, record: LogRecord) -> str:
        """Format a logging record.

        Parameters
        ----------
        record : LogRecord
            Log record emitted by the logging framework.

        Returns
        -------
        str
            Formatted log message.
        """
        icon = self.ICONS.get(record.levelname, "❓")
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        message = (
            f"{icon} {record.levelname.ljust(8)} {timestamp} "
            f"[{record.filename}:{record.lineno}] "
            f"🚀 {record.getMessage()}"
        )
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return message


def create_logger(log_path: str | Path) -> logging.Logger:
    """Create a file and console logger.

    Parameters
    ----------
    log_path : str or Path
        Path to the log file.

    Returns
    -------
    logging.Logger
        Configured logger keyed by ``log_path``.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = CustomFormatter()

    # ------------------------
    # 1. File handler
    # ------------------------
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # ------------------------
    # 2. Console handler
    # ------------------------
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False

    return logger
