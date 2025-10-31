from .training_params import TrainingParams, LightningConfig, TorchConfig, DataConfig, TransformConfig, ModelConfig, CallbackConfig, WandbConfig
from typing import Optional, List

class TrainingParams(TrainingParams):
    lightning: LightningConfig = LightningConfig(
        name="lightning",
        params={
            "max_epochs": 2400,
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
            "img_root": ["hexray25_train/ds/ds1/images/", "hexray25_train/ds/ds2/images/", "hexray25_train/ds/ds3/images/", "hexray25_train/ds/ds4/images/"],
            "mask_root": ["hexray25_train/ds/ds1/defect_mask_all_as_one/", "hexray25_train/ds/ds2/defect_mask_all_as_one/", "hexray25_train/ds/ds3/defect_mask_all_as_one/", "hexray25_train/ds/ds4/defect_mask_all_as_one/"],
            "train_txt_loc": ["hexray25_train/ds/ds1/train_all.txt", "hexray25_train/ds/ds2/train_all.txt", "hexray25_train/ds/ds3/train_all.txt", "hexray25_train/ds/ds4/train_all.txt"],
            "val_txt_loc": ["hexray25_train/ds/ds1/test_all.txt", "hexray25_train/ds/ds2/test_all.txt", "hexray25_train/ds/ds3/test_all.txt", "hexray25_train/ds/ds4/test_all.txt"],
            "batch_size": 2,
            "transforms":TransformConfig(name="vjt_all_defects_v34"),
            "img_height": 2048,
            "img_width": 2048,
        },
        
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
                    image_size=2048,
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
                scheduler={"type": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts", "T_0": 10, "T_mult": 2},
                infer_type={"type": "patch_inferer", "patch_size": (2048, 2048), "batch_size": 2}
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
            "name": "vjt-upernet-v34-all-defect",
            "save_dir": "./lightning",
        },
    )


vjt_v34_all_defect = TrainingParams()


# old config - ignore
# seed: 1984

# dl:
#   __class_fullname__: vjt.data_loaders.PLTxtLoader
#   img_root: ["/root/vjt-data-2/ds1/images/", "/root/vjt-data-2/ds2/images/", "/root/vjt-data-2/ds3/images/", "/root/vjt-data-2/ds4/images/"]
#   mask_root: ["/root/vjt-data-2/ds1/defect_mask_all_as_one/", "/root/vjt-data-2/ds2/defect_mask_all_as_one/", "/root/vjt-data-2/ds3/defect_mask_all_as_one/", "/root/vjt-data-2/ds4/defect_mask_all_as_one/"]
#   train_txt_loc: ["/root/vjt-data-2/ds1/train.txt", "/root/vjt-data-2/ds2/train.txt", "/root/vjt-data-2/ds3/train.txt", "/root/vjt-data-2/ds4/train.txt"]
#   val_txt_loc: ["/root/vjt-data-2/ds1/test.txt", "/root/vjt-data-2/ds2/test.txt", "/root/vjt-data-2/ds3/test.txt", "/root/vjt-data-2/ds4/test.txt"]
#   test_txt_loc: ["/root/vjt-data-2/ds1/test.txt", "/root/vjt-data-2/ds2/test.txt", "/root/vjt-data-2/ds3/test.txt", "/root/vjt-data-2/ds4/test.txt"]
#   batch_size: 4


# infer_type:
#   patch_size:
#     - 2048
#     - 2048
#   batch_size: 2

# model:
#   type: vjtx.network.HuggingFaceModel
#   network: vjtx.network.UperNetForSemanticSegmentation
#   backbone:
#     type: transformers.models.convnextv2.configuration_convnextv2.ConvNextV2Config
#     num_channels: 3
#     patch_size: 4 
#     hidden_sizes: 
#       - 64
#       - 128
#       - 256
#       - 512
#     depths: 
#       - 2
#       - 2
#       - 6
#       - 2
#     num_stages: 4 
#     image_size: 1024 #Do we need to change this? 
#     out_features: 
#     - stage1 
#     - stage2
#     - stage3
#     - stage4
#   model_config:
#     type: vjtx.network.UperNetConfigCustom
#     num_labels: 1 
#     hidden_size: 256
#     use_auxiliary_head: False
#     backbone_config: backbone
#     temperature: false 

# optimizer:
#   type: torch.optim.AdamW
#   lr: 0.001

# trainer:
#   type: lightning.Trainer
#   accelerator: "gpu"
#   devices: 8
#   max_epochs: 2400
#   strategy: auto
#   gradient_clip_val: 5.0
#   num_sanity_val_steps: 2
#   log_every_n_steps: 5

# scheduler:
#   type: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
#   T_0: 10
#   T_mult: 2

# checkpoint_callback:
#   type: lightning.pytorch.callbacks.ModelCheckpoint
#   monitor: val_epoch_iou
#   verbose: True
#   mode: max
#   save_top_k: 5
#   filename: "{epoch}-{step}-{val_epoch_iou:.3f}"

# lr_callback:
#    type: lightning.pytorch.callbacks.LearningRateMonitor
#    logging_interval: "step"
  
# tqdm_callback:
#    type: lightning.pytorch.callbacks.TQDMProgressBar
#    refresh_rate: 4

# logger:
#   type: lightning.pytorch.loggers.WandbLogger
#   project: "vjtx-defects-v34"
#   save_dir: "./lighting_dir"
#   name: "v1-base-exp1"

# pretrained_weight_loc: "/root/vjt-data-ablation/lighting_dir/vjtx-defects/ymljv7c2/checkpoints/epoch=625-step=217848-val_epoch_iou=0.281.ckpt"

# train_aug:
#   transform:
#     - __class_fullname__: vjtx.tfsm.PIL2Tensor
#     - __class_fullname__: vjtx.tfsm.CustomColorJitter
#       brightness: 0.5
#       contrast: 0.5
#       saturation: 0.5
#       hue: 0.5
#     - __class_fullname__: vjtx.tfsm.CustomRandomHorizontalFlip
#       p: 0.5
#     - __class_fullname__: vjtx.tfsm.CustomRandomVerticalFlip
#       p: 0.5
#     - __class_fullname__: vjtx.tfsm.CustomRandomRotation90
#       p: 0.5
#     - __class_fullname__: vjtx.tfsm.CustomResize
#       height: 4096
#       width: 4096
#     - __class_fullname__: vjtx.tfsm.Int32Normalize
#     - __class_fullname__: vjtx.tfsm.CropNonEmptyMaskIfExists
#       height: 2048
#       width: 2048

# val_aug:
#   transform:
#     - __class_fullname__: vjtx.tfsm.PIL2Tensor
#     - __class_fullname__: vjtx.tfsm.CustomResize
#       height: 4096
#       width: 4096
#     - __class_fullname__: vjtx.tfsm.Int32Normalize