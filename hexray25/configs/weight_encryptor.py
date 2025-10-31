from pydantic import BaseModel
from typing import Dict, Any, List

class ModelsConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class EDConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class WeightEncryptorConfig(BaseModel):
    ed_config: EDConfig = EDConfig(
            name="paths",
            params={
            "aes_key_path_raw": "hexray25/encryption/aes_key.bin",
            "aes_key_path_encrypted": "hexray25/encryption/aes_key_encrypted.bin",
            "rsa_key_path_private": "hexray25/encryption/rsa_key_private.pem",
            "rsa_key_path_public": "hexray25/encryption/rsa_key_public.pem",
            "signature_path_raw": "hexray25/encryption/signature.txt",
            "signature_path_encrypted": "hexray25/encryption/signature_encrypted.bin"
        }
    )
    
    models_config: List[ModelsConfig] = [ModelsConfig(
        name="bg",
        params={
            "model_path_raw": "hexray25/pretrained_weights/bg_epoch=306-step=103766-val_epoch_iou=0.992.ckpt",
            "model_path_encrypted": "hexray25/pretrained_weights/bg_epoch=306-step=103766-val_epoch_iou=0.992.enc"
        }

    ), 
    ModelsConfig(
        name="all_defects_v32",
        params={
            "model_path_raw": "hexray25/pretrained_weights/all_defects_v32_epoch=786-step=273876-val_epoch_iou=0.294.ckpt",
            "model_path_encrypted": "hexray25/pretrained_weights/all_defects_v32_epoch=786-step=273876-val_epoch_iou=0.294.enc"
        }
    ),
    ModelsConfig(
        name="fmmd_v32",
        params={
            "model_path_raw": "hexray25/pretrained_weights/fmmd_v32_epoch=1229-step=318570-val_epoch_iou=0.306.ckpt",
            "model_path_encrypted": "hexray25/pretrained_weights/fmmd_v32_epoch=1229-step=318570-val_epoch_iou=0.306.enc"
        }
    ), 
    ModelsConfig(
        name="fmmd_v34",
        params={
            "model_path_raw": "hexray25/pretrained_weights/v34/fmmd_epoch=2277-step=590002-val_epoch_iou=0.352.ckpt",
            "model_path_encrypted": "hexray25/pretrained_weights/v34/fmmd_0352.enc"
        }
    ),
    ModelsConfig(
        name="all_defects_v34",
        params={
            "model_path_raw": "hexray25/pretrained_weights/v34/all_defects_epoch=2375-step=826848-val_epoch_iou=0.383.ckpt",
            "model_path_encrypted": "hexray25/pretrained_weights/v34/all_defects_0383.enc"
        }
    )
    ]
        
