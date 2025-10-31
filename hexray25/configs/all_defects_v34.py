from pydantic import BaseModel
from typing import Dict, Any, List, Tuple

class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}
    model_weights: str = ""


class TorchConfig(BaseModel):
    name: str
    value: str


class TransformConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class InferenceParams(BaseModel):
    patch_size_default: Tuple[int, int] = (2048, 2048)
    magnification: float = 0.5
    batch_size: int = 2


class AllDefectsV34InferenceParams(BaseModel):
    torch_config: List[TorchConfig] = [
        TorchConfig(name="float32_matmul_precision", value="medium")
    ]

    transforms: TransformConfig = TransformConfig(name="vjt_fmmd_v34")

    model_config = {"type": "hexray25.models.upernet.HuggingFaceModel", 
                "network": "hexray25.models.upernet.UperNetForSemanticSegmentation", 
                "backbone": dict(
                    type="transformers.models.convnextv2.configuration_convnextv2.ConvNextV2Config",
                    num_channels=3,
                    patch_size=4,
                    hidden_sizes=[64, 128, 256, 512],
                    depths=[2, 2, 6, 2],
                    num_stages=4,
                    image_size=1024,
                    out_features=["stage1", "stage2", "stage3", "stage4"],
                ), 
                "model_config": dict(
                    type="hexray25.models.upernet.UperNetConfigCustom",
                    num_labels=1,
                    hidden_size=256,
                    use_auxiliary_head=False,
                    backbone_config=None,
                    temperature=False,
                )}

    inference_params: InferenceParams = InferenceParams()

    model: ModelConfig = ModelConfig(
        name="upernet",
        params=dict(hparams=dict(model=model_config)),
    )
    

all_defects_v34_defaults = AllDefectsV34InferenceParams()
    
    