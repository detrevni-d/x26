import torch 
import numpy as np
import cv2
from hexray25.configs.bg import BGInferenceParams
from hexray25.registries import ModelRegistry, TransformRegistry
from typing import Dict, Any
from torch.nn import functional as F

class BGModelInferenceV1(torch.nn.Module):
    def __init__(self, params: BGInferenceParams, weights: Dict[str, Any], device: str):
        super().__init__()
        self.model = ModelRegistry.get(params.model.name, **params.model.params)
        self.model.load_state_dict(weights["state_dict"])
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.params = params
        self.transforms = TransformRegistry.get(params.transforms.name)
        self.val_tfsm = self.transforms.get_val_transforms(self.params.transforms.params["img_height"], self.params.transforms.params["img_width"])
        
    def __call__(self, img: np.ndarray, pred_threshold: float=0.5)->np.ndarray:
        timg = self.val_tfsm({"image": img})["image"]
        timg = timg.to(self.device)
        pred = self.model(timg[None])
        if isinstance(pred, list): 
            if len(pred) != 1: raise ValueError("list should be of length 1")
            pred = pred[0]
        pred = F.logsigmoid(pred).exp()
        pred = pred.detach().cpu().numpy()[0][0]
        pred = ((pred > pred_threshold)*255).astype(np.uint8)
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return pred
    
