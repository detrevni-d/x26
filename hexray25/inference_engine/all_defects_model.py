import torch 
import numpy as np
from hexray25.configs.all_defects_v32 import AllDefectsV32InferenceParams
from hexray25.configs.all_defects_v34 import AllDefectsV34InferenceParams
from hexray25.registries import ModelRegistry, TransformRegistry
from monai.inferers import AvgMerger, PatchInferer, SlidingWindowSplitter
from typing import Dict, Any
from torch.nn import functional as F

class AllDefectsV32ModelInference(torch.nn.Module):
    def __init__(self, params: AllDefectsV32InferenceParams, weights: Dict[str, Any], device: str):
        super().__init__()
        #self.params = params #<-May not be necessary?
        self.model = ModelRegistry.get(params.model.name, **params.model.params)
        self.model.load_state_dict(weights["state_dict"])
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.transforms = TransformRegistry.get(params.transforms.name)
        #self.val_tfsm = self.transforms.get_inference_transforms(self.params.transforms.params["img_height"], self.params.transforms.params["img_width"])
        
    def __call__(self, img: np.ndarray)->np.ndarray:
        #timg = self.val_tfsm({"image": img})["image"]
        tfsm = self.transforms.get_inference_transforms(img_height=img.shape[0], img_width=img.shape[1])
        timg = tfsm({"image": img})["image"]

        timg = timg.to(self.device)
        pred = self.model(timg[None])
        if isinstance(pred, list): 
            if len(pred) != 1: raise ValueError("list should be of length 1")
            pred = pred[0]
        pred = F.logsigmoid(pred).exp()
        pred = pred.detach().cpu().numpy()[0][0]
        pred = ((pred > 0.5)*255).astype(np.uint8)
        return pred
    


class AllDefectsV34ModelInference(torch.nn.Module):
    def __init__(self, params: AllDefectsV34InferenceParams, weights: Dict[str, Any], device: str):
        super().__init__()
        self.model = ModelRegistry.get(params.model.name, **params.model.params)
        self.model.load_state_dict(weights["state_dict"])
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.transforms = TransformRegistry.get(params.transforms.name, **params.transforms.params)
        self.patch_inferer = PatchInferer(splitter=SlidingWindowSplitter(patch_size=params.inference_params.patch_size_default, overlap=0, pad_mode=None),
                                            merger_cls=AvgMerger,
                                            match_spatial_shape=True, 
                                            batch_size=params.inference_params.batch_size,
                                        )
        #TODO: Add patch_inferer separately for common sizes 
        
    def __call__(self, img: np.ndarray)->np.ndarray:
        tfsm = self.transforms.get_inference_transforms(img_height=img.shape[0]*4, img_width=img.shape[1]*4)
        
        if img.shape[0] == 1024 and img.shape[1] == 1024:
            patch_inferer = self.patch_inferer  
        else:   
            patch_inferer = PatchInferer(splitter=SlidingWindowSplitter(patch_size=(img.shape[0]*2, img.shape[1]*2)), 
                                            merger_cls=AvgMerger,
                                            match_spatial_shape=True, 
                                            batch_size=1
                                        )
        patch_inferer.merger_kwargs = {"device": self.device}
        timg = tfsm({"image": img})["image"]
        timg = timg.to(self.device)
        pred = patch_inferer(timg[None], self.model)
        if isinstance(pred, list): 
            if len(pred) != 1: raise ValueError("list should be of length 1")
            pred = pred[0]
        pred = F.logsigmoid(pred).exp()
        pred = pred.detach().cpu().numpy()[0][0]
        pred = ((pred > 0.5)*255).astype(np.uint8)
        return pred