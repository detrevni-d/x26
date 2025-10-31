import torch 
import torchvision
import random
from PIL import Image
from torchvision import transforms
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2.functional import InterpolationMode
from torchvision.transforms.v2 import ColorJitter
from pydantic import BaseModel


class RandomScale:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, inputs):
        img = inputs["image"]
        orig_height, orig_width = img.shape[-2:]
        
        # Get random scale factor
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        new_height = int(orig_height * scale_factor)
        new_width = int(orig_width * scale_factor)

        # Scale image
        img = F.resize(img, (new_height, new_width), InterpolationMode.BILINEAR)
        
        if "mask" in inputs.keys():
            mask = inputs["mask"]
            mask = F.resize(mask, (new_height, new_width), InterpolationMode.NEAREST)
            return {"image": img, "mask": mask}
        
        return {"image": img}

class CustomResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, inputs):
        img = inputs["image"]
        img = F.resize(img, (self.height, self.width), InterpolationMode.BILINEAR)
        if "mask" in inputs.keys():
            mask = inputs["mask"]
            mask = F.resize(mask, (self.height, self.width), InterpolationMode.NEAREST)
            return {"image": img, "mask": mask}
        return {"image": img}


class CropNonEmptyMaskIfExists:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, inputs):
        if "image" not in inputs or "mask" not in inputs:
            raise ValueError("Inputs must contain both 'image' and 'mask'")
        img, mask = inputs["image"], inputs["mask"]
        if (len(img.shape) != 3) or (len(mask.shape) != 3):
            raise ValueError("Image and mask must be a 3D tensor")
        # we will use pil to tensor and it create a channel as first channel by default
        # so we need to convert it to c,h,w format
        mask_height, mask_width = mask.shape[1:]
        if mask.any():
            mask_ = mask.sum(axis=0)
            non_zero_yx = torch.argwhere(mask_)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, self.width - 1)
            y_min = y - random.randint(0, self.height - 1)
            x_min = torch.clip(x_min, 0, mask_width - self.width)
            y_min = torch.clip(y_min, 0, mask_height - self.height)
        else:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)
        
        x_max = x_min + self.width
        y_max = y_min + self.height

        img = img[:, y_min:y_max, x_min:x_max]
        mask = mask[:, y_min:y_max, x_min:x_max]
        return {"image": img, "mask": mask}


class CustomColorJitter:
    def __init__(self, brightness, contrast, saturation, hue, p=0.5):
        self.tf1 = ColorJitter(brightness, contrast, None, None)
        self.tf2 = ColorJitter(None, None, saturation, hue)
        self.p = p
    def __call__(self, inputs):
        if "image" not in inputs or "mask" not in inputs:
            raise ValueError("Inputs must contain both 'image' and 'mask'")
        img, mask = inputs["image"], inputs["mask"]
        if random.random() < self.p:
            img = self.tf1(img)
        if random.random() < self.p:
            img = self.tf2(img)
        
        return {"image": img, "mask": mask}

class CustomColorJitter2:
    def __init__(self, brightness=0.3, contrast=0.3, p=0.5):
        self.tf = ColorJitter(brightness, contrast)
        self.p = p

    def __call__(self, inputs):
        if "image" not in inputs or "mask" not in inputs:
            raise ValueError("Inputs must contain both 'image' and 'mask'")
        img, mask = inputs["image"], inputs["mask"]
        if random.random() < self.p:
            img = self.tf(img)
        return {"image": img, "mask": mask}



class PIL2Tensor:
    def __init__(self):
        pass

    def __call__(self, inputs):
        img= inputs["image"]
        img = F.to_image(img).float()
        # if img.shape[0] == 3:
        #     img = img[0][None]
        if img.shape[0] != 3:
            # create a 3 channel image
            img = img.repeat(3, 1, 1)
        
        if "mask" in inputs:
            mask = F.to_image(inputs["mask"])
            return {"image": img, "mask": mask}
        return {"image": img}


class NumPy2Tensor:
    def __init__(self, channels=3):
        self.channels = channels

    def __call__(self, inputs):
        img = inputs["image"].copy()
        if len(img.shape) == 2:
            img = img[None]
        img = torch.from_numpy(img)
        if img.shape[0] != self.channels:
            img = img.repeat(self.channels, 1, 1)
        if "mask" in inputs:
            mask = inputs["mask"]
            return {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask)}
        return {"image": img}


class ByteImage2Tensor:
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass 



class CustomRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, inputs):
        if "image" not in inputs or "mask" not in inputs:
            raise ValueError("Inputs must contain both 'image' and 'mask'")
        img, mask = inputs["image"], inputs["mask"]
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return {"image": img, "mask": mask}


class CustomRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, inputs):
        if "image" not in inputs or "mask" not in inputs:
            raise ValueError("Inputs must contain both 'image' and 'mask'")
        img, mask = inputs["image"], inputs["mask"] 
        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return {"image": img, "mask": mask}

class Int32Normalize:
    def __init__(self):
        pass

    def __call__(self, inputs):
        img = inputs["image"]
        img = img.float()
        imgx = (img - img.min())/(img.max() - img.min() + 1e-6)
        if "mask" in inputs:
            mask = inputs["mask"]
            mask = mask / 255.0
            return {"image": imgx, "mask": mask}
        return {"image": imgx}

class CustomRandomRotation90:
    def __init__(self, p=0.5): 
        self.degrees = 90
        self.p = p

    def __call__(self, inputs):
        if "image" not in inputs or "mask" not in inputs:
            raise ValueError("Inputs must contain both 'image' and 'mask'")
        img, mask = inputs["image"], inputs["mask"]
        if random.random() < self.p:
            img = F.rotate(img, self.degrees)
            mask = F.rotate(mask, self.degrees)
        return {"image": img, "mask": mask}


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.norm = torchvision.transforms.Normalize(mean, std)

    def __call__(self, inputs):
        img = inputs["image"]
        img = self.norm(img)
        if "mask" in inputs:
            mask = inputs["mask"]
            return {"image": img, "mask": mask}
        return {"image": img}


class VJTSegTransforms(BaseModel):
    def get_train_transforms(self)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            CustomRandomHorizontalFlip(p=0.5),
            CustomRandomVerticalFlip(p=0.5),
            CustomRandomRotation90(p=0.5),
            RandomScale(scale_range=(1.0, 1.6)),
            Int32Normalize(),
            CropNonEmptyMaskIfExists(1024, 1024),
        ])

    def get_val_transforms(self)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomResize(1024, 1024),
            Int32Normalize(),
        ])


class VJTBGSegTransforms(BaseModel):
    
    def get_train_transforms(self, img_height, img_width)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            CustomRandomHorizontalFlip(p=0.5),
            CustomRandomVerticalFlip(p=0.5),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])

    def get_val_transforms(self, img_height, img_width)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])

    def get_inference_transforms(self, img_height, img_width)->transforms.Compose:
        return transforms.Compose([
            NumPy2Tensor(channels=3),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])

class VJTAllDefectsV32SegTransforms(BaseModel):
    def get_train_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            # CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            CustomColorJitter2(brightness=0.3, contrast=0.2),
            CustomRandomHorizontalFlip(p=0.5),
            CustomRandomVerticalFlip(p=0.5),
            CustomRandomRotation90(p=0.5),
            RandomScale(scale_range=(1.0, 1.6)),
            Int32Normalize(),
            CropNonEmptyMaskIfExists(img_height, img_width),
        ])

    def get_inference_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            NumPy2Tensor(channels=3),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])

    def get_val_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])
    
class VJTFMMDV32SegTransforms(BaseModel):
    def get_train_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            CustomRandomHorizontalFlip(p=0.5),
            CustomRandomVerticalFlip(p=0.5),
            CustomRandomRotation90(p=0.5),
            RandomScale(scale_range=(1.0, 1.6)),
            Int32Normalize(),
            CropNonEmptyMaskIfExists(img_height, img_width),
        ])

    def get_inference_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            NumPy2Tensor(channels=3),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])
    
    def get_val_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])

class VJTFMMDV34SegTransforms(BaseModel):
    def get_train_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            CustomRandomHorizontalFlip(p=0.5),
            CustomRandomVerticalFlip(p=0.5),
            CustomRandomRotation90(p=0.5),
            Int32Normalize(),
            CustomResize(img_height*2, img_width*2),
            CropNonEmptyMaskIfExists(img_height, img_width),
        ])

    def get_val_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomResize(img_height*2, img_width*2),
            Int32Normalize(),
        ])
    
    def get_inference_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            NumPy2Tensor(channels=3),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])


class VJTAllDefectsV34SegTransforms(BaseModel):
    def get_train_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            # CustomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            CustomColorJitter2(brightness=0.3, contrast=0.3),
            CustomRandomHorizontalFlip(p=0.5),
            CustomRandomVerticalFlip(p=0.5),
            CustomRandomRotation90(p=0.5),
            Int32Normalize(),
            CropNonEmptyMaskIfExists(img_height, img_width),
        ])

    def get_val_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            PIL2Tensor(),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])
    
    def get_inference_transforms(self, img_height=1024, img_width=1024)->transforms.Compose:
        return transforms.Compose([
            NumPy2Tensor(channels=3),
            CustomResize(img_height, img_width),
            Int32Normalize(),
        ])