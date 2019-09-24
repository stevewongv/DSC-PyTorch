import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
import PIL.Image as Image
from randomcrop import RandomHorizontallyFlip

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.root = '../SBU-shadow/SBUTrain4KRecoveredSmall/'
        self.imgs = open(self.dataset).readlines()
        self.file_num = len(self.imgs)

        self.hflip = RandomHorizontallyFlip()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])

    def __len__(self):
        return self.file_num * 100

    def __getitem__(self, index):
        image_path,label_path = self.imgs[index % self.file_num][:-1].split(' ')
        image = Image.open(self.root + image_path).convert('RGB').resize((400,400))
        label = Image.open(self.root + label_path).convert('L').resize((400,400))

        image,label = self.hflip(image,label)

        label = np.array(label,dtype='float32') / 255.0
        if len(label.shape) > 2:
            label = label[:,:,0]
        
        image_nom = self.trans(image)
        label = np.array([label])

        sample = {'O': image_nom,'B':label,'image':np.array(image,dtype='float32').transpose(2,0,1)/255}
        return sample



class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.root = '../SBU-shadow/SBU-Test/'
        self.imgs = open(self.root + 'SBU.txt').readlines()
        self.file_num = len(self.imgs)

        self.hflip = RandomHorizontallyFlip()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])

    def __len__(self):
        return self.file_num 

    def __getitem__(self, index):

        image_path,label_path = self.imgs[index % self.file_num][:-1].split(' ')
        image = Image.open(self.root + image_path).convert('RGB').resize((400,400))
        label = Image.open(self.root + label_path).convert('L').resize((400,400))

        label = np.array(label,dtype='float32') / 255.0
        if len(label.shape) > 2:
            label = label[:,:,0]
        
        image_nom = self.trans(image)

        sample = {'O': image_nom,'B':label,'image':np.array(image)}

        return sample
