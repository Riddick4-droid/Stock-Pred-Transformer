import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from src.exceptions import StockTransformerException

# Default log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str = "stock_transformer",level: str = "INFO",log_file: str = "app.log",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Create and return a logger with console and file handlers.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : str
        File name inside LOG_DIR.
    max_bytes : int
        Maximum file size before rotation.
    backup_count : int
        Number of backup files to keep.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding handlers twice
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    file_path = LOG_DIR / log_file
    file_handler = RotatingFileHandler(
        file_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# logger instance for direct import/ or comment out so that you can import the function directly and initilaize in the scripts
logger = setup_logger()