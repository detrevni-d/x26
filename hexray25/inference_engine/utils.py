from dataclasses import dataclass
from typing import Annotated, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from functools import partial
from loguru import logger
import cv2
import matplotlib
import numpy as np
import torch
import json
import yaml
import pydoc
import re
from PIL import Image
from io import BytesIO
import fastcore.all as fc

def load_img(img: Union[str, np.ndarray, bytes]):
        logger.info("Load image")
        if isinstance(img, str) or isinstance(img, fc.Path): img = np.asarray(Image.open(img))
        elif isinstance(img, np.ndarray): img = img
        elif isinstance(img, bytes): img = np.asarray(Image.open(BytesIO(img)))
        #img = np.asarray(Image.open(BytesIO(base64.b64decode(img))))
        else: raise NotImplementedError("only loc, bytes np.ndarray are allowed")
        return img

def mask_overlays(img, mask, alpha=1.0, beta=0.4):
    # image is h, w and mask is h,w
    # num_labels, labels = cv2.connectedComponents((mask * 255.0).astype(np.uint8), connectivity=8)
    colors = np.asarray(matplotlib.colormaps.get_cmap("tab10").colors)[1]
    mask_colors = colors.copy()
    mask_colors = (mask_colors * 255).astype(np.uint8)

    ## get mask
    seg_mask = mask[:, :, None].repeat(3, axis=2)
    seg_mask = np.where(seg_mask == 1, mask_colors, seg_mask)

    if img.dtype == np.int32 or img.dtype == np.uint16:
        fimg = np.uint8((img - img.min()) / (img.max() - img.min() + 1e-8) * 255)
        fimg = fimg[:, :, None].repeat(3, axis=2)
    else:
        raise NotImplementedError("only int32 is supported for now")

    overlayed = cv2.addWeighted(fimg, alpha, seg_mask.astype(np.uint8), beta, 0)
    return fimg, overlayed

@dataclass
class RLEStore:
    shape: Annotated[Tuple[int], 2]
    rle: str

@dataclass
class VJTOutput:
    image: Union[np.ndarray]
    parts: Union[np.ndarray, RLEStore]
    only_defects: Union[np.ndarray, RLEStore]
    defect: Union[np.ndarray, RLEStore]
    image_overlay: Optional[np.ndarray]

def mask2rle(img):
    """
    Efficient implementation of mask2rle, from @paulorzp
    --
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    Source: https://www.kaggle.com/xhlulu/efficient-mask2rle
    """
    pixels = img.T.flatten()
    pixels = np.pad(pixels, ((1, 1),))
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1024, 1024)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def pad_to_multiple_of_32(image):
    # Get original shape
    original_shape = image.shape[:2]
    
    # Calculate padding for both axes
    pad_h = (32 - image.shape[0] % 32) % 32
    pad_w = (32 - image.shape[1] % 32) % 32
    
    # Pad to make both dimensions multiples of 32
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    else:
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')

    
    # Determine which axis is longer after padding
    long_axis = 0 if padded_image.shape[0] > padded_image.shape[1] else 1
    short_axis = 1 - long_axis
    
    # Calculate additional padding for short axis
    long_size = padded_image.shape[long_axis]
    short_size = padded_image.shape[short_axis]
    additional_pad = long_size - short_size
    
    # Create padding tuple for final padding
    final_padding = [(0, 0), (0, 0), (0, 0)] if len(image.shape) ==3 else [(0, 0), (0, 0)] 
    final_padding[short_axis] = (0, additional_pad)
    
    # Apply final padding
    final_padded_image = np.pad(padded_image, final_padding, mode='constant')
    
    return final_padded_image, original_shape

def unpad_image(padded_image, original_shape):
    return padded_image[:original_shape[0], :original_shape[1]]


def pad_and_unpad_array(input_array, target_multiple=1024):
    # Check if the input is 2D or 3D
    is_2d = len(input_array.shape) == 2

    # Get the original dimensions
    if is_2d:
        original_height, original_width = input_array.shape
    else:
        original_height, original_width = input_array.shape[:2]

    # Calculate the new dimensions (multiples of target_multiple)
    new_height = ((original_height - 1) // target_multiple + 1) * target_multiple
    new_width = ((original_width - 1) // target_multiple + 1) * target_multiple

    # Calculate padding
    pad_height = new_height - original_height
    pad_width = new_width - original_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Pad the array
    if is_2d:
        padded_array = np.pad(input_array, 
                              ((pad_top, pad_bottom), (pad_left, pad_right)), 
                              mode='constant')
    else:
        padded_array = np.pad(input_array, 
                              ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                              mode='constant')

    # Function to remove padding
    def unpad_array(padded):
        is_2d = len(padded.shape) == 2
        if is_2d:
            return padded[pad_top:pad_top+original_height, 
                          pad_left:pad_left+original_width]
        else:
            return padded[pad_top:pad_top+original_height, 
                          pad_left:pad_left+original_width, :]

    return padded_array, unpad_array

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

if __name__ == "__main__":
    t0, _ = pad_and_unpad_array(np.ones((756, 1542)), 1024)
    t1, _ = pad_and_unpad_array(np.ones((1432, 1411)), 1024)
    t2, _ = pad_and_unpad_array(np.ones((2048, 2048)), 1024)
    t3, _ = pad_and_unpad_array(np.ones((2048+12, 2048+13)), 1024)
    t4, _ = pad_and_unpad_array(np.ones((2400, 2400)), 1024)
    t5, unpad_func = pad_and_unpad_array(np.ones((2047, 2400)), 1024)
    print([i.shape for i in [t0, t1, t2, t3, t4, t5]])
    print(unpad_func(t5).shape)