import datetime
import os

from .training_params import TrainingParams, LightningConfig, TorchConfig, DataConfig, TransformConfig, ModelConfig, CallbackConfig, WandbConfig
from pydantic import BaseModel
from typing import Optional, List
from hexray25.configs.dataset_config import DEFAULT_ROOT, DEFAULT_STORE_ROOT
from hexray25.configs.dataset_config import WANDB_STUB, REPO_BASE
from lightning.pytorch.accelerators import find_usable_cuda_devices

ds_base_path = os.path.join(REPO_BASE,'hexray25_train','ds')
available_gpus = find_usable_cuda_devices()
print(f'available_gpus={available_gpus}\n===\n')

if len(available_gpus)==4:
    # 4x4090
    wandb_info = {
        "devices":4, 
        "batch_size": 5,# Use batchsize of 5 on each gpu so effective BS=(4*5)
        "accumulate_grad_batches": 13, #to keep effective BS = 4*5*13=260
    }
elif len(available_gpus)==1:
    # single 4090 GPU
    wandb_info = {
        "devices":1, #all available GPUs
        "batch_size": 5,# For single 4090 use 5. For 4x4090 using 16
        "accumulate_grad_batches": 52,# to keep effective BS = 1*5*52=260
    }

class TrainingParams(TrainingParams):
    lightning: LightningConfig = LightningConfig(
        name="lightning",
        params={
            "max_epochs": 1000,
            "accelerator": "gpu",
            "devices": wandb_info["devices"],
            "strategy": "auto",
            "sync_batchnorm": True,
            "accumulate_grad_batches": wandb_info["accumulate_grad_batches"],#<-
            # "ckpt_path": "lightning/hexraynet/e8qaiend/checkpoints/vjt-upernet-epoch\=5-step\=876-val_epoch_iou\=0.000.ckpt"
        },
    )
    torch_config: List[TorchConfig] = [
        TorchConfig(name="float32_matmul_precision", value="medium")
    ]
    data: DataConfig = DataConfig(
        name="vjt",
        params={
            "img_root": [os.path.join(ds_base_path,'ds1/images/'), 
                         os.path.join(ds_base_path,'ds2/images/'), 
                         os.path.join(ds_base_path,'ds3/images/'), 
                         os.path.join(ds_base_path,'ds4/images/')],
            "mask_root": [os.path.join(ds_base_path,'ds1/defect_mask_all_as_one/'), 
                          os.path.join(ds_base_path,'ds2/defect_mask_all_as_one/'), 
                          os.path.join(ds_base_path,'ds3/defect_mask_all_as_one/'), 
                          os.path.join(ds_base_path,'ds4/defect_mask_all_as_one/')],
            "train_txt_loc": [os.path.join(ds_base_path,'ds1/train_all.txt'), 
                              os.path.join(ds_base_path,'ds2/train_all.txt'), 
                              os.path.join(ds_base_path,'ds3/train_all.txt'), 
                              os.path.join(ds_base_path,'ds4/train_all.txt')],
            "val_txt_loc": [os.path.join(ds_base_path,'ds1/test_all.txt'), 
                            os.path.join(ds_base_path,'ds2/test_all.txt'), 
                            os.path.join(ds_base_path,'ds3/test_all.txt'), 
                            os.path.join(ds_base_path,'ds4/test_all.txt')],
            "batch_size": wandb_info["batch_size"],
            "transforms":TransformConfig(name="vjt_all_defects_v32"),
            "img_height": 1024,
            "img_width": 1024,
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
                "save_top_k": 50,
                "filename": "vjt-upernet-{epoch}-{step}-{val_epoch_iou:.3f}",
            },
        ),
        CallbackConfig(name="lr_monitor", params={"logging_interval": "step"}),
        CallbackConfig(name="tqdm_progress_bar", params={"refresh_rate": 4}),
        CallbackConfig(name="memory_cleanup", params={"cleanup_frequency": 10}),
    ]
    wandb: Optional[WandbConfig] = WandbConfig(
        name="wandb",
        params={
            "project": "hexraynet",
            "name": f"vjt-v32-all-defect-{WANDB_STUB}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "save_dir": "./lightning",
            "config": {'model_config':model_config,'more': wandb_info},            
            "resume": "allow",
        },
    )


vjt_v32_all_defect = TrainingParams()
