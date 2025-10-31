from pydantic import BaseModel
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from typing import Optional, List

from hexray25.configs.dataset_config import DSNAME, DEFAULT_ROOT, DEFAULT_STORE_ROOT
from hexray25.coco_utils import COCO


class DatasetCreatorVJT(BaseModel):
    root: Path = DEFAULT_ROOT
    store_root: Path = DEFAULT_STORE_ROOT
    ds_name: str
    fmmd: bool = False
    consider_bg: bool = False
    store_masked_images: bool = False

    def create_directories(self, store_folder: str) -> None:
        """Create necessary directories for storing data."""
        print(f"***self.store_root = {self.store_root} ***")
        (self.store_root / self.ds_name / store_folder).mkdir(parents=True, exist_ok=True)
        (self.store_root / self.ds_name / "images").mkdir(parents=True, exist_ok=True)

    def process_image(self, img: str, ds: COCO, store_folder: str) -> None:
        """Process a single image and its annotations.
        
        Writes the image and[if necessary] a mask (depends on the data set)
        """
        img_path = self.root / "images" / img
        store_path = self.store_root / self.ds_name / "images" / img
        
        # Loads binary BG masks
        if self.consider_bg:
            bg_path = self.store_root / "parts" / "masks" / img
            if not bg_path.exists():
                print(f"bg img for {img} not found")
                return
            bg_img = np.asarray(Image.open(bg_path))
            bg_img = np.where(bg_img > 0, 1, 0)

        mask_path = self.store_root / self.ds_name / store_folder / img

        if store_path.exists() and mask_path.exists():
            return

        # anns is a list of dicts?
        _, anns = ds.loadimgAnns(img_name=img, root=self.root)
        # filter out categories of annotations that are not needed
        anns = [i for i in anns if i["category_id"] != 8]  # Porosity hard to do is not required
        anns = [i for i in anns if i["category_id"] not in [9, 10, 11]]  # Porosity hard to do is not required
        if self.fmmd:
            anns = [i for i in anns if i["category_id"] == 1]  # FMMD is required

        if len(anns) == 0:
            print(f"anns for {img} is empty")
            mask = np.zeros(Image.open(img_path).size)
            return None 
        else:
            mask = np.concatenate([np.expand_dims(ds.annToMask(i), 0) for i in anns]).sum(0)
            if self.consider_bg:
                mask[mask > 0] = 1
                mask = np.logical_and(mask, bg_img)
                mask = np.uint8(mask)
            mask[mask > 0] = 255

        mask = Image.fromarray(np.uint8(mask))
        if not mask_path.exists():
            mask.save(mask_path)

        if not self.store_masked_images:
            if not store_path.exists():
                shutil.copy(img_path, store_path)
        else:
            if not store_path.exists():
                if bg_path.exists():
                    img = Image.open(img_path)
                    bg_mask = Image.open(bg_path)
                    img = np.array(img)
                    bg_mask = np.array(bg_mask)
                    img = np.where(bg_mask == 255, img, 0)
                    img = Image.fromarray(img)
                    img.save(store_path)
                else:
                    shutil.copy(img_path, store_path)

    def create_dataset(self) -> None:
        """Main method to create the dataset."""
        if self.ds_name not in DSNAME:
            raise ValueError(f"Invalid dataset name. Choose from: {list(DSNAME.keys())}")

        store_folder = "defect_mask_fmmd" if self.fmmd else "defect_mask_all_as_one"
        
        # Load dataset
        ds = COCO.for_vjt((self.root / "annotations" / DSNAME[self.ds_name]).as_posix())
        print(f"Processing {len(ds.imgname)} images")

        self.create_directories(store_folder)

        for img in tqdm(ds.imgname, desc="Processing images"):
            self.process_image(img, ds, store_folder)
