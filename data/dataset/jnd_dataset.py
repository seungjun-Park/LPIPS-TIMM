import os.path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import cv2, glob

class JNDDataset(Dataset):
    def __init__(self, dataroot, load_size=64):
        self.root = dataroot
        self.load_size = load_size

        self.p0_paths = glob.glob(f'{self.root}/*/p0/*.png')
        self.p0_paths = sorted(self.p0_paths)

        self.p1_paths = glob.glob(f'{self.root}/*/p1/*.png')
        self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list += [transforms.ToTensor(),
                           transforms.Resize(load_size),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.same_paths = glob.glob(f'{self.root}/*/same/*.npy')
        self.same_paths = sorted(self.same_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = cv2.imread(p0_path, cv2.IMREAD_COLOR)
        p0_img_ = cv2.cvtColor(p0_img_, cv2.COLOR_BGR2RGB)
        p0_img = self.transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = cv2.imread(p1_path, cv2.IMREAD_COLOR)
        p1_img_ = cv2.cvtColor(p1_img_, cv2.COLOR_BGR2RGB)
        p1_img = self.transform(p1_img_)

        same_path = self.same_paths[index]
        same_img = np.load(same_path).reshape((1,1,1,)) # [0,1]

        same_img = torch.FloatTensor(same_img)

        return p0_img, p1_img, same_img
    
    def __len__(self):
        return len(self.p0_paths)
