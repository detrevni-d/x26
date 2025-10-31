from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class Image(BaseModel):
    id: int
    file_name: str
    height: int
    width: int
    coco_url: Optional[str]
    date_captured: Union[int, Optional[str]]
    flickr_url: Optional[str]
    license: Union[int, Optional[str]]

    class Config:
        arbitrary_types_allowed = True


class Annotation(BaseModel):
    image_id: int
    id: int
    category_id: int
    bbox: Optional[List[float]]
    area: Optional[float]
    segmentation: Optional[Union[Dict, List[List[float]]]]
    iscrowd: Optional[int]
    value: Optional[str] = None
    name: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class Categories(BaseModel):
    id: int 
    name: str 
    supercategory: str = "" 


class Licenses(BaseModel):
    name: str 
    id: int 
    url: str

class CocoInfo(BaseModel):
    contributor: str= ""
    date_created: str= ""
    description: str=  ""
    url: str= "" 
    version: str= ""
    year: str= ""


class CocoDataset(BaseModel):
    images: List[Image]
    categories: List[Categories]
    annotations: List[Annotation]
    licenses: Optional[List[Licenses]]
    info: Optional[List[CocoInfo]]

    class Config:
        arbitrary_types_allowed = True
        # Changed from orm_mode to from_attributes to fix a pydantic V2 warning
        from_attributes = True #orm_mode=True