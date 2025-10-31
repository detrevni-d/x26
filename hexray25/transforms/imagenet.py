from pydantic import BaseModel
from torchvision import transforms


class ImagenetTransforms(BaseModel):
    input_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    def get_train_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def get_val_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(int(self.input_size * 1.14)),  # 256 for 224
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
