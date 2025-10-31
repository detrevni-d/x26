import torch
import lightning as pl
import fastcore.all as fc
from PIL import Image
from typing import List
from loguru import logger
from torch.utils.data import DataLoader
from hexray25.registries.transforms import TransformRegistry

class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_root, mask_root, txt_file=None, transforms=None):
        self.img_root = fc.Path(img_root)
        self.mask_root = fc.Path(mask_root)
        if txt_file is not None:
            self.req_imgs = fc.L([i.strip() for i in open(txt_file).readlines()])
        else:
            self.req_imgs = None
        self.imgs = fc.L(self.img_root.glob("*.png"))
        self.imgs = [i for i in self.imgs if (self.mask_root / i.name).exists()]
        if self.req_imgs is not None:
            self.imgs = fc.L([i for i in self.imgs if i.name in self.req_imgs])
        print(f"Dataset: {len(self.imgs)} images")
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = {}
        x["image"] = Image.open(self.img_root / self.imgs[idx].name)
        x["mask"] = Image.open(self.mask_root / self.imgs[idx].name)
        if self.transforms is not None:
            x = self.transforms(x)
        return x["image"], x["mask"]


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.combined_data = []
        for dataset in datasets:
            self.combined_data.extend([(i, dataset) for i in range(len(dataset))])

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        original_idx, dataset = self.combined_data[idx]
        return dataset[original_idx]


class VJTDataModule(pl.LightningDataModule):
    def __init__(
        self, img_root: List[str], mask_root: List[str], train_txt_loc: List[str], val_txt_loc: List[str], batch_size: int = 32, transforms=None, img_height: int = 2048, img_width: int = 2048, num_workers: int = 4
    ):
        super().__init__()
        self.img_root = img_root
        self.mask_root = mask_root
        self.batch_size = batch_size
        self.train_txt_loc = train_txt_loc
        self.val_txt_loc = val_txt_loc
        self.num_workers = num_workers
        logger.info(f"Initializing VJT dataset")

        # Lazy import of transforms to avoid circular dependency
        if transforms is None:
            # from hexray25.registries.transforms import TransformRegistry

            # transforms = TransformRegistry.get("vjt_seg")
            raise ValueError("transforms are not initialized")
        self.transforms =TransformRegistry.get(transforms.name)
        self.transform_train = self.transforms.get_train_transforms(img_height, img_width)
        self.transform_val = self.transforms.get_val_transforms(img_height, img_width)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CombinedDataset([
                Dataset(img_root, mask_root, txt_loc, self.transform_train) for img_root, mask_root, txt_loc in zip(self.img_root, self.mask_root, self.train_txt_loc)
            ])
            self.val_dataset = CombinedDataset([
                Dataset(img_root, mask_root, txt_loc, self.transform_val) for img_root, mask_root, txt_loc in zip(self.img_root, self.mask_root, self.val_txt_loc)
            ])

            logger.debug(
                f"Found {len(self.train_dataset)} training images and {len(self.val_dataset)} validation images"
            )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,# This drops the last batch if it's smaller than batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )