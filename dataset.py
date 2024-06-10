import os
from typing import Any
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch # try

class Bdd100kDataset(Dataset) :
    def __init__(self , image_dir , mask_dir , transform = None) :
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) :
        return len(self.images)
    
    def __getitem__(self , index) :
        img_path = os.path.join(self.image_dir , self.images[index])
        
        mask_path = os.path.join(self.mask_dir , self.images[index].replace(".jpg" , "_mask.gif")) # for TUTORIAL
        # mask_path = os.path.join(self.mask_dir , self.images[index].replace(".jpg" , ".png")) # for CVPR seg
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        mask = np.array(Image.open(mask_path).convert("L") , dtype = np.float32) # 0.0 , 255.0 # for TUTORIAL
        # mask = np.array(Image.open(mask_path).convert("L"))
        mask[mask == 255.0] = 1.0 # for TUTORIAL (only 0.0 & 255.0)

        if self.transform is not None :
            augmentations = self.transform(image = image , mask = mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            # mask = torch.from_numpy(np.array(mask))
            # mask = mask.type(torch.LongTensor)
            # mask = torch.max(mask , dim = 2)[0] # try

        return image , mask