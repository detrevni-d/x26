from typing import Dict, Type

from loguru import logger

from ..transforms.imagenet import ImagenetTransforms
from ..transforms.vjt_seg import VJTSegTransforms, VJTBGSegTransforms, VJTAllDefectsV32SegTransforms, VJTFMMDV32SegTransforms, VJTFMMDV34SegTransforms, VJTAllDefectsV34SegTransforms
from .base import BaseRegistry


class TransformRegistry(BaseRegistry):
    _registry: Dict[str, Type] = {}


# Register available transforms
TransformRegistry.register("imagenet", ImagenetTransforms)
TransformRegistry.register("vjt_seg", VJTSegTransforms)
TransformRegistry.register("vjt_bg_seg", VJTBGSegTransforms)
TransformRegistry.register("vjt_all_defects_v32", VJTAllDefectsV32SegTransforms)
TransformRegistry.register("vjt_fmmd_v32", VJTFMMDV32SegTransforms)
TransformRegistry.register("vjt_fmmd_v34", VJTFMMDV34SegTransforms)
TransformRegistry.register("vjt_all_defects_v34", VJTAllDefectsV34SegTransforms)

logger.info("Transform registry initialized with available transforms")
