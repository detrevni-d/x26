from typing import Dict, Type

import lightning as pl
from hexray25.datasets.imagenette import ImagenetteDataModule
from hexray25.datasets.vjt import VJTDataModule
from loguru import logger

from .base import BaseRegistry


class DataRegistry(BaseRegistry):
    _registry: Dict[str, Type[pl.LightningDataModule]] = {}


# Register available datasets
DataRegistry.register("imagenette", ImagenetteDataModule)
DataRegistry.register("vjt", VJTDataModule)
logger.info("Data registry initialized with available datasets")
