import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CustomImageDataset(Dataset):
    def __init__(self, data, transforms=None, cache=None):
        self.data = data
        self.transforms = transforms
        self.num_classes = 5
        self.cache_image = cache
        # self.imgdir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        def get_image_cache(idx):
            img_path = self.data.iloc[idx, 0]
            # img_path = img_path.replace("../Data", "D:\DATA\해충분류")


            if img_path in self.cache_image:
                return img_path, self.cache_image[img_path]

            else:
                img_array = np.fromfile(img_path, np.uint8)

                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                image = cv2.cvtColor(image, cv2.IMREAD_COLOR)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
                image = image.astype(np.float32) / 255.0
                # print(image.dtype, image.min(), image.max())

                if self.transforms is not None:
                    image = self.transforms(image=image)['image']

                self.cache_image[img_path] = torch.as_tensor(image, dtype=torch.float32)

                return img_path, self.cache_image[img_path]

        img_path, image = get_image_cache(idx)
        label = self.data.iloc[idx, 1]
        # if img_path in self.cache_image:
        #     print(f"cache_image hit for: {img_path}")
        # else:
        #     print(f"cache_image miss for: {img_path}. Adding to cache_image.")

        return {'image': image, 'label': label, 'path': img_path}

def transform():
    return A.Compose([
        A.Resize(224,224),
        ToTensorV2()])

def make_loader(data, batch_size, shuffle, cache):
    return DataLoader(
        dataset = CustomImageDataset(data, transforms=transform(), cache=cache),
        batch_size = batch_size,
        shuffle = shuffle
    )

# def make_loader(data, batch_size, shuffle):
#     return DataLoader(
#         dataset = CustomImageDataset(data, transforms=transform()),
#         batch_size = batch_size,
#         shuffle = shuffle
#     )