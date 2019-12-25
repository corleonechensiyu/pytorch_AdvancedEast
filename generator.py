
import numpy as np
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import cfg


class custom_dataset(data.Dataset):
    def __init__(self, img_path): #train_image_dir_name
        super(custom_dataset, self).__init__()
        self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_filename = self.img_files[index].strip().split('/')[-1]

        gt_file = os.path.join(cfg.data_dir,
                               cfg.train_label_dir_name,
                               img_filename[:-4] + '_gt.npy')
        y=np.load(gt_file)
        img = Image.open(self.img_files[index])
        transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        return transform(img),torch.Tensor(y).permute(2,0,1)




