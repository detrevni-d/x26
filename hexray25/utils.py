import copy
import json
import pydoc
import re
import torch 
import yaml

import fastcore.all as fc

from functools import partial
from hexray25.coco_utils import COCO
from loguru import logger
from pathlib import Path
from PIL import Image
import PIL
from typing import Any, Dict, Optional, Union

# TODO: had trouble importing this function into a notebook
# had trouble using fc. Solution might be to restart the notebook?
def LoadBGAnnotations(vjt_data_path):
    """ Load annotations to do background training
    """
    #root = fc.Path("/home/ubuntu/foundations/vjt-data/")
    root = fc.Path(vjt_data_path)
    
    images_loc = fc.L((root / "images").glob("*.png"))
    print(f"total images: {len(images_loc)}")
    annots1 = COCO.for_vjt(root / "annotations/phase6_background_separation_27apr24.json")
    annots2 = COCO.for_vjt(root / "annotations/background_seperation_ds3_ds4_bg_jan12_2025_fixed.json")
    annots3_stamp = COCO.for_vjt(root / "annotations/full_bg_backup_may2025.json")

    # Per Adith on 2025/7/30 via MS-Teams: need to clean up full_bg_backup_may2025.json:
    # Clone the original object safely
    annots3 = copy.deepcopy(annots3_stamp)
    
    # Only strip prefix if anchor pattern is present
    anchor = "ADE.RAW.88DC3"
    
    for img in annots3.imgs.values():
        fname = img['file_name']
        if anchor in fname:
            parts = fname.split("_", 2)
            if len(parts) >= 3:
                img['file_name'] = parts[-1]

    return annots1, annots2, annots3

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

def load_yaml(path):
    with open(path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)
    return hparams

def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('I', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def thumbnail(img, size=256):
    if not isinstance(img, PIL.Image.Image): img = Image.fromarray(img)
    w, h = img.size
    ar = h/w 
    return img.resize((size, int(size*ar)))

def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)

def locate_cls(transforms: dict, return_partial=False):
    name = transforms["__class_fullname__"]
    targs = {k: v for k, v in transforms.items() if k != "__class_fullname__"}
    try:
        if return_partial:
            transforms = partial(pydoc.locate(name), **targs)
        else:
            transforms = pydoc.locate(name)(**targs)
    except Exception as e:
        logger.error(f"Cannot load {name}. Error: {str(e)}")
    return transforms

def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result

def state_dict_from_disk(
    file_path: Union[Path, str], rename_in_layers: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.

    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:
        state_dict = rename_layers(state_dict, rename_in_layers)
    if "hyper_parameters" in checkpoint.keys():
        return state_dict, checkpoint["hyper_parameters"]
    return state_dict