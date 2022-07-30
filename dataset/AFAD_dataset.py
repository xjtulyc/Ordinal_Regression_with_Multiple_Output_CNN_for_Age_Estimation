import glob

import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class AFAD(Dataset):
    def __init__(self, train=False):
        super(AFAD, self).__init__()
        self.path = 'dataset/tarball/AFAD-Full'
        self.gender = {'male': '111', 'female': '112'}
        self.age_range = (15, 72 + 1)
        self.transform = None
        self.num_imgs = len(glob.glob(self.path + '/*/*/*.jpg'))
        self.img_list = glob.glob(self.path + '/*/*/*.jpg')
        self.is_train = train
        self.idx = np.array([i for i in range(self.__len__())], dtype=int)
        self.Image_Transform()

    def Image_Transform(self):
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((60, 60)),
                transforms.Normalize(0, 1),
                transforms.RandomCrop(60),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((60, 60)),
            ])
        pass

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        idx = idx
        img = Image.open(self.img_list[idx])
        temp_list = self.img_list[idx].split('/')
        age = int(temp_list[-3])
        label = torch.zeros(72 - 15 + 1, 2)
        label[:age - 15] = torch.tensor([1, 0])
        label[age - 15:] = torch.tensor([0, 1])
        # img = (transforms.ToTensor()(img)-0.5)*2
        img = self.transform(img)
        return img, label, age
