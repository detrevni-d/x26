from typing import Dict, Type

import lightning as pl
from hexray25.models.resnet18 import Resnet18
from hexray25.models.upernet import UperNetSegmenter
from loguru import logger

from .base import BaseRegistry


class ModelRegistry(BaseRegistry):
    _registry: Dict[str, Type[pl.LightningModule]] = {}


# Register available models
ModelRegistry.register("resnet18", Resnet18)
ModelRegistry.register("upernet", UperNetSegmenter)

logger.info("Model registry initialized with available models")
