import fastcore.all as fc 
import numpy as np 
import time
import json

from PIL import Image
from collections import Counter, defaultdict
from pycocotools.coco import COCO


@fc.patch_to(COCO, as_prop=True)
def imgid2name(self):
    return {i['id']: i["file_name"] for _, i in self.imgs.items()}

@fc.patch_to(COCO, as_prop=True)
def imgname2id(self):
    return {i['file_name']: i["id"] for _, i in self.imgs.items()}

@fc.patch_to(COCO, as_prop=True)
def imgname(self):
    return fc.L(self.imgname2id.keys())

@fc.patch_to(COCO, as_prop=True)
def catId2name(self):
    return {t["id"]:t["name"] for t in self.loadCats(self.getCatIds())}

@fc.patch_to(COCO, as_prop=True)
def catname2Id(self):
    return {t["name"]:t["id"] for t in self.loadCats(self.getCatIds())}

@fc.patch_to(COCO)
def print_stats(self):
    cats = self.loadCats(self.getCatIds())
    nms= [cat['name'] for cat in cats]
    print(f"total annotations:: {len(self.anns)}")
    print(f"total categories:: {len(nms)}")
    print(f'Defects categories::')
    for n, cat in enumerate(nms): print(f"   {n}-{cat}")

@fc.patch_to(COCO)
def _loadimg(self, img_id=None, img_name=None):
    if (img_id is None) and (img_name is None): raise ValueError("both img_id and img_name cannot be None")
    if img_id is None: img_id = self.imgname2id[img_name]
    img = self.loadImgs(ids=[img_id])[0]
    return img
    
@fc.patch_to(COCO)
def loadimgAnns(self, img_id=None, img_name=None, root=None, catIds=[]):
    img = self._loadimg(img_id=img_id, img_name=img_name)
    annIds = self.getAnnIds(imgIds=img['id'], iscrowd=None, catIds=catIds)
    anns = self.loadAnns(ids=annIds)
    if root is not None:
        loc = root/"images"/img["file_name"] if "base_name" not in img.keys() else \
              root/img["base_name"]/"images"/img["file_name"]
        I = np.asarray(Image.open(loc))
        return I, anns 
    return anns

@fc.patch_to(COCO)
def countannots_by_imgid(self, img_id=None, img_name=None):
    img = self._loadimg(img_id=img_id, img_name=img_name)
    annIds = self.getAnnIds(imgIds=img["id"], iscrowd=None, catIds=[])
    anns = self.loadAnns(ids=annIds)
    return Counter(fc.L([self.catId2name[i["category_id"]] for i in  anns]))    

def remove_stuff(dataset):
    dataset["corrupted_annotations"] = [i for i in dataset["annotations"] if len(i["segmentation"]) ==0]
    dataset["annotations"] = [i for i in dataset["annotations"] if len(i["segmentation"]) !=0]
    print(f"Removed annotations: {len(dataset['corrupted_annotations'])}")
    
    imgs_with_annots = np.unique([i["image_id"] for i in dataset["annotations"]])
    dataset["images_without_annots"] = [i for i in dataset["images"] if i["id"] not in imgs_with_annots]
    dataset["images"] = [i for i in dataset["images"] if i["id"] in imgs_with_annots]
    print(f"Removed images: {len(dataset['images_without_annots'])}")
    return dataset

@fc.patch_to(COCO)
def __init__(self, annotation_file):
    self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
    self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
    if not annotation_file == None:
        print('loading annotations into memory...')
        tic = time.time()
        dataset = json.load(open(annotation_file, 'r')) if isinstance(annotation_file, str) else annotation_file
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.dataset = dataset
        self.createIndex()
        if "corrupted_annotations" in self.dataset: self.corrupted_annotations = self.dataset["corrupted_annotations"]
        if "images_without_annots" in self.dataset: self.images_without_annots = self.dataset["images_without_annots"]
            
@fc.patch_to(COCO, cls_method=True)
def for_vjt(cls, annotation_file):
    if not annotation_file == None:
        print('read and processing annotations for corruptions')
        tic = time.time()
        dataset = json.load(open(annotation_file, 'r'))
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        annotation_file = remove_stuff(dataset)
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
    return cls(annotation_file = annotation_file)
