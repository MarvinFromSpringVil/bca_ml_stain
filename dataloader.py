from PIL import Image
import torch
from torch import nn
import os
from glob import glob
import random 

from torch.utils.data import Dataset, DataLoader

class DeepStainDataset(Dataset):
    def __init__(self, root_dir, mode='dapi', transforms=None):
        self.root_dir = root_dir
        self.transform = transform

        self.x_images = glob(os.path.join(os.path.join(root_dir, 'x'), '*.png'))

        if mode == 'dapi':
            self.y_images = glob(os.path.join(os.path.join(root_dir, 'y1'), '*.png'))
        elif mode == 'lap2':
            self.y_images = glob(os.path.join(os.path.join(root_dir, 'y2'), '*.png'))
        else:
            raise NotImplementedError('Unsupported mode: ', mode)

        assert len(self.x_images) == len(self.y_images)
    
        self.transform = transform

    def __len__(self):
        return len(self.x_images)

    def __getitem__(self, idx):
        x = Image.open(self.x_images[idx])
        y = Image.open(self.y_images[idx])

        if random.random() < 0.5: 
            z = torch.ones(1, 28, 28) 
        else:
            z = torch.zeros(1, 28, 28) 

        x = self.transform(x)
        y = self.transform(y)
        
        sample = {'x': x, 'y': y, 'z': z}
        return sample

def get_dataloader(root_dir, batch_size, mode, transforms, shuffle=False):
    dataset = DeepStainDataset(root_dir, mode=mode, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

if __name__ == '__main__':
    from torchvision import transforms

    ROOT_DIR = './dataset_root'
    TRANSFORMS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = DeepStainDataset(ROOT_DIR, TRANSFORMS)

    print(len(dataset)) 

    for elem in dataset:
        print(elem['x'].shape)
        print(elem['y'].shape)
        print(elem['z'].shape)
        break 
    