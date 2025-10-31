from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CallbackConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class ModelConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class TorchConfig(BaseModel):
    name: str
    value: str


class TransformConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class DataConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}
    transforms: TransformConfig = TransformConfig(
        name="imagenet", params={"input_size": 224}
    )


class WandbConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class LightningConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class TrainingParams(BaseModel):
    lightning: LightningConfig = LightningConfig(
        name="lightning",
        params={
            "max_epochs": 100,
            "accelerator": "gpu",
            "devices": 8,
            "strategy": "ddp",
            "precision": 16,
            "sync_batchnorm": True,
            # gradient_clip_val=0.5,
            # accumulate_grad_batches=1
        },
    )
    torch_config: List[TorchConfig] = [
        TorchConfig(name="float32_matmul_precision", value="medium")
    ]
    data: DataConfig = DataConfig(
        name="imagenette",
        params={
            "root": "./data/imagenette",
            "download": False,
            "batch_size": 1024,
        },
        transforms=TransformConfig(name="imagenet", params={"input_size": 224}),
    )
    model: ModelConfig = ModelConfig(
        name="resnet50",
        params={"num_classes": 10, "learning_rate": 1e-5},
    )
    callbacks: List[CallbackConfig] = [
        CallbackConfig(
            name="model_checkpoint",
            params={
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "filename": "resnet-imagenette-{epoch:02d}-{val_loss:.2f}",
            },
        ),
        CallbackConfig(name="lr_monitor", params={"logging_interval": "step"}),
        CallbackConfig(
            name="early_stopping",
            params={"monitor": "val_loss", "patience": 5, "mode": "min"},
        ),
        CallbackConfig(name="tqdm_progress_bar", params={"refresh_rate": 4}),
    ]
    wandb: Optional[WandbConfig] = WandbConfig(
        name="wandb",
        params={
            "project": "hexraynet",
            "name": "resnet-imagenette",
            "save_dir": "./lightning",
        },
    )


classification_defaults = TrainingParams()
