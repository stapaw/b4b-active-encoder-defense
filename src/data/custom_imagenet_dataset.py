import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class CustomImageNetDataset(Dataset):
    def __init__(self, root_dir, num_images=50000, transform=None):
        self.root_dir = root_dir
        self.num_images = num_images
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        for class_dir in self.classes:
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                class_images = os.listdir(class_path)
                class_images = [os.path.join(class_dir, img) for img in class_images]
                random.shuffle(class_images)
                self.images.extend(class_images[:num_images//len(self.classes)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.images[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.classes.index(self.images[index].split("/")[0])
        return img, label
