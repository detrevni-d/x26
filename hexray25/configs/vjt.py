from .training_params import TrainingParams, LightningConfig, TorchConfig, DataConfig, TransformConfig, ModelConfig, CallbackConfig, WandbConfig
from pydantic import BaseModel
from typing import Optional, List

class TrainingParams(TrainingParams):
    lightning: LightningConfig = LightningConfig(
        name="lightning",
        params={
            "max_epochs": 600,
            "accelerator": "gpu",
            "devices": 1,
            "strategy": "auto",
            "sync_batchnorm": True,
        },
    )
    torch_config: List[TorchConfig] = [
        TorchConfig(name="float32_matmul_precision", value="medium")
    ]
    data: DataConfig = DataConfig(
        name="vjt",
        params={
            "img_root": ["/root/vjt-data-2/ds1/images/", "/root/vjt-data-2/ds2/images/", "/root/vjt-data-2/ds3/images/", "/root/vjt-data-2/ds4/images/"],
            "mask_root": ["/root/vjt-data-2/ds1/defect_mask_all_as_one/", "/root/vjt-data-2/ds2/defect_mask_all_as_one/", "/root/vjt-data-2/ds3/defect_mask_all_as_one/", "/root/vjt-data-2/ds4/defect_mask_all_as_one/"],
            "train_txt_loc": ["/root/vjt-data-2/ds1/train.txt", "/root/vjt-data-2/ds2/train.txt", "/root/vjt-data-2/ds3/train.txt", "/root/vjt-data-2/ds4/train.txt"],
            "val_txt_loc": ["/root/vjt-data-2/ds1/test.txt", "/root/vjt-data-2/ds2/test.txt", "/root/vjt-data-2/ds3/test.txt", "/root/vjt-data-2/ds4/test.txt"],
            "batch_size": 4,
        },
        transforms=TransformConfig(name="vjt_seg"),
    )
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
    
    model: ModelConfig = ModelConfig(
        name="upernet",
        params=dict(hparams=dict(model=model_config, 
                optimizer={"type": "torch.optim.AdamW", "lr": 1e-4},
                scheduler={"type": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts", "T_0": 10, "T_mult": 2}
        ))
                
    )
    callbacks: List[CallbackConfig] = [
        CallbackConfig(
            name="model_checkpoint",
            params={
                "monitor": "val_epoch_iou",
                "mode": "max",
                "save_top_k": 10,
                "filename": "vjt-upernet-{epoch}-{step}-{val_epoch_iou:.3f}",
            },
        ),
        CallbackConfig(name="lr_monitor", params={"logging_interval": "step"}),
        CallbackConfig(name="tqdm_progress_bar", params={"refresh_rate": 4}),
    ]
    wandb: Optional[WandbConfig] = WandbConfig(
        name="wandb",
        params={
            "project": "hexraynet",
            "name": "vjt-upernet",
            "save_dir": "./lightning",
        },
    )


vjt_defaults = TrainingParams()
