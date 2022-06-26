import torch
import numpy as np
from torch.utils.data import Dataset
from data_processing import data_processing, VOC_parser
from PIL import Image

"""
The three funtion you see here is necessary for all the custom dataset
"""

class fruit_dataset(Dataset):
    def __init__(self, root, transforms):
        """
        just init some variable
        """
        self.root = root
        self.transform = transforms
        self.imgs = data_processing(root)["imgs"]
        self.xml = data_processing(root)["xml"]
        self.transforms = transforms
        self.classes = ["background","apple","banana","orange"]
    def __getitem__(self, index):
        """
        When you loop the train/test loader, the program will come here
        to ask for data
        so it mush has a argument called index or idx or anyname.
        This function must return a image, and the target dictionary
        becase the pytorch model needs these.
        """
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        boxes,temp_labels = VOC_parser(self.xml[index])
        
        labels = []
        for label in temp_labels:
            labels.append(self.classes.index(label))
        
        boxes = torch.tensor(boxes,dtype=torch.float32)
        labels = torch.tensor(labels,dtype=torch.int64)
        image_id = torch.tensor([index])
        # area = width * height
        # boxes = (xmin,ymin,xmax,ymax)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Just keep it 0. I don't know exactly why
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        img = img.type(torch.float32)
        """
        You have to normalize the data because neural network likes
        evenly distributed data

        In plain english. If the different between the data is small,
        that means their standard deviation is low. The data is evenly distributed.
        Our image pixel from 0 to 255. The range is way to wide.
        we need to norm to 0-1

        And this is called normalize.
        
        """
        img /= 255.0
        return img, target
    def __len__(self):
        return len(self.imgs)