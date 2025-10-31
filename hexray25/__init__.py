import sys
from pathlib import Path

import torch
from loguru import logger

__version__ = "0.1.0"

logger.remove()
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure console logging with custom format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Configure file logging with more detailed format
logger.add(
    log_dir / "hexraynet_{time}.log",
    rotation="12:00",  # New file at noon
    retention="30 days",
    compression="zip",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    backtrace=True,
    diagnose=True,
    enqueue=True,  # Thread-safe logging
)

# Add error logging to separate file
logger.add(
    log_dir / "errors_{time}.log",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    backtrace=True,
    diagnose=True,
    enqueue=True,
)

logger.info("Logging system initialized")

is_cuda_available = torch.cuda.is_available()
logger.info(f"CUDA is available: {is_cuda_available}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"PyTorch is using device: {device}")

if torch.cuda.is_available():
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")
