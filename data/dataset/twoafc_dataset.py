import os.path
import torchvision.transforms as transforms
from torch.utils.data import Dataset
# from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import cv2, glob


class TwoAFCDataset(Dataset):
    def __init__(self, 
                   dataroot, 
                   load_size=64, 
                   train=True):
        self.root = dataroot
        self.load_size = load_size

        if train:
            split = 'train'
        else:
            split = 'val'

        # image directory
        self.ref_paths = glob.glob(f'{self.root}/{split}/*/ref/*.png')
        self.ref_paths = sorted(self.ref_paths)

        self.p0_paths = glob.glob(f'{self.root}/{split}/*/p0/*.png')
        self.p0_paths = sorted(self.p0_paths)

        self.p1_paths = glob.glob(f'{self.root}/{split}/*/p1/*.png')
        self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list += [transforms.ToTensor(),
                           transforms.Resize(load_size),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.judge_paths = glob.glob(f'{self.root}/{split}/*/judge/*.npy')
        self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = cv2.imread(p0_path, cv2.IMREAD_COLOR)
        p0_img_ = cv2.cvtColor(p0_img_, cv2.COLOR_BGR2RGB)
        p0_img = self.transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = cv2.imread(p1_path, cv2.IMREAD_COLOR)
        p1_img_ = cv2.cvtColor(p1_img_, cv2.COLOR_BGR2RGB)
        p1_img = self.transform(p1_img_)

        ref_path = self.ref_paths[index]
        ref_img_ = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        ref_img_ = cv2.cvtColor(ref_img_, cv2.COLOR_BGR2RGB)
        ref_img = self.transform(ref_img_)

        judge_path = self.judge_paths[index]
        # judge_img = (np.load(judge_path)*2.-1.).reshape((1,1,1,)) # [-1,1]
        judge = np.load(judge_path).reshape((1,1,1,)) # [0,1]

        judge = torch.from_numpy(judge)

        return p0_img, p1_img, ref_img, judge

    def __len__(self):
        return len(self.p0_paths)
