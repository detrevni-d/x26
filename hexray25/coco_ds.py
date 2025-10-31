import fastcore.all as fc
from PIL import Image as PILImage
import numpy as np

CAT_IDS = [1, 2, 3, 4, 5, 6, 7]

class DS2:
    def __init__(self, img_root, coco_annot, img_ids):
        self.img_root = img_root
        self.annots = coco_annot
        self.imgids2coco_file = {k:v for v, i in enumerate(self.annots) for k in i.imgname}
        if img_ids is None:
            self.img_ids = [j for i in self.annots for j in i.imgname]
        else:
            self.img_ids = img_ids
        print("total images: ", len(self.img_ids))
        self.cat_ids = fc.L(self.annots[0].catId2name.keys())
    
    def __len__(self): return len(self.img_ids)
    
    def __getitem__(self, img_id):
        """ Allows the ability to access an element list-like eg. ds[n]"""
        p={}
        p["img_name"] = self.img_ids[img_id]
        p["ds"] = self.imgids2coco_file[p["img_name"]]
        annot = self.annots[p["ds"]]
        p["img_id"] = annot.imgname2id[p["img_name"]]
        #print(fc.Path(self.img_root)/p['img_name'])
        p["pil_image"] = PILImage.open(fc.Path(self.img_root)/p['img_name'])
        
        annIds = [j for i in CAT_IDS for j in annot.getAnnIds(imgIds=[p["img_id"]], catIds=[i], iscrowd=None)]
        anns = [annot.loadAnns(ann) for ann in annIds]
        p["labels"] = fc.L([i[0]["category_id"] for i in anns])
        p["masks"] = [annot.annToMask(ann[0]) for ann in anns]
        bbox = np.asarray([ann[0]["bbox"] for ann in anns])
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
        p["bbox"] = bbox
        return p 
    
    def only_foreground_defects(self, img_id, bg_mask):
        #1, 1 = 1
        #1, 0 = 0
        #0, 1 = 0
        #0, 0 = 0
        p = self[img_id]
        idx = [n for n, i in enumerate(p["masks"]) if (bg_mask*i).sum() != 0]
        p["masks"] = [p["masks"][i] for i in idx]
        p["labels"] = [p["labels"][i] for i in idx]
        p["bbox"] = [p["bbox"][i] for i in idx]
        return p