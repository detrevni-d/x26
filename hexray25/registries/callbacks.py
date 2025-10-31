from typing import Dict, Type

from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
import torch
import gc
from loguru import logger

from .base import BaseRegistry

class MemoryCleanupCallback(Callback):
    """Callback to periodically free CPU and GPU memory during training."""

    def __init__(self, cleanup_frequency: int = 100, log_usage: bool = False):
        """
        Args:
            cleanup_frequency (int): Clean up memory every N training steps
            log_usage (bool): Whether to log memory usage before/after cleanup
        """
        self.cleanup_frequency = cleanup_frequency
        self.log_usage = log_usage
        self.step_count = 0

    def _cleanup(self, msg: str):
        """Perform CPU and GPU memory cleanup with optional logging."""
        gc.collect()  # free Python objects
        
        if torch.cuda.is_available():
            before_alloc = torch.cuda.memory_allocated() / 1024**2 if self.log_usage else None
            before_reserved = torch.cuda.memory_reserved() / 1024**2 if self.log_usage else None

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if self.log_usage:
                after_alloc = torch.cuda.memory_allocated() / 1024**2
                after_reserved = torch.cuda.memory_reserved() / 1024**2
                logger.debug(
                    f"[{msg}] GPU Mem Alloc: {before_alloc:.2f} MB → {after_alloc:.2f} MB | "
                    f"Reserved: {before_reserved:.2f} MB → {after_reserved:.2f} MB"
                )
            else:
                logger.debug(f"[{msg}] CPU + GPU memory cleaned")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Clean memory every N training batches."""
        self.step_count += 1
        if self.cleanup_frequency > 0 and self.step_count % self.cleanup_frequency == 0:
            self._cleanup(f"After train batch {batch_idx}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Clean memory after validation batch."""
        self._cleanup(f"After validation batch {batch_idx}")

    def on_train_epoch_end(self, trainer, pl_module):
        """Clean memory at the end of every epoch."""
        self._cleanup("After epoch end")

class CallbackRegistry(BaseRegistry):
    _registry: Dict[str, Type[Callback]] = {}


# Register built-in callbacks
CallbackRegistry.register("model_checkpoint", ModelCheckpoint)
CallbackRegistry.register("lr_monitor", LearningRateMonitor)
CallbackRegistry.register("early_stopping", EarlyStopping)
CallbackRegistry.register("tqdm_progress_bar", TQDMProgressBar)
CallbackRegistry.register("memory_cleanup", MemoryCleanupCallback)
logger.info("Callback registry initialized with built-in callbacks")
