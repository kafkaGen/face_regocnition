import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms as T

from settings.config import Config


class FaceRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = os.listdir(data_path)

    def __len__(self):
        return len(self.classes) * 2

    def __getitem__(self, index):
        if index >= len(self.classes):
            index -= len(self.classes)
            class_path = os.path.join(self.data_path, str(index))
            neg_classes = list(set(self.classes) - set(str(index)))
            neg_class = np.random.choice(neg_classes, 1)[0]
            neg_class_path = os.path.join(self.data_path, str(neg_class))

            img1 = cv.imread(os.path.join(class_path, np.random.choice(os.listdir(class_path), 1)[0]))
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
            img2 = cv.imread(os.path.join(neg_class_path, np.random.choice(os.listdir(neg_class_path), 1)[0]))
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            label = 0
        else:
            class_path = os.path.join(self.data_path, str(index))
            pair = np.random.choice(os.listdir(class_path), 2, replace=False)
            img1 = cv.imread(os.path.join(class_path, pair[0]))
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
            img2 = cv.imread(os.path.join(class_path, pair[1]))
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
            label = 1

        img1 = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img2 = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            img1 = T.Normalize(Config.mean, Config.std)(img1)
            img2 = T.Normalize(Config.mean, Config.std)(img2)

        return img1, img2, label
