from typing import Dict, Type

import lightning as pl
from loguru import logger

from hexray25.runners.base_trainer import BaseTrainer

from .base import BaseRegistry


class RunnerRegistry(BaseRegistry):
    _registry: Dict[str, Type[pl.LightningModule]] = {}


# Register available models
RunnerRegistry.register("base", BaseTrainer)

logger.info("Runner registry initialized with available runners")
