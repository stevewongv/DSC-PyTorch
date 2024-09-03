import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
import PIL.Image as Image
from randomcrop import RandomHorizontallyFlip

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.root = '/home/zhxing/Datasets/SRD_inpaint4shadow_fix/'
        # self.root = '/home/zhxing/Datasets/ISTD+/'

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
        line = self.imgs[index % self.file_num].strip()
        parts = line.split()
        image_path, label_path = parts[0], parts[1]

        image = cv2.imread(self.root + image_path)
        label = cv2.imread(self.root + label_path)

        # Convert to LAB color space
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        label_lab = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)

        # Resize images
        image_lab = cv2.resize(image_lab, (512, 512))
        label_lab = cv2.resize(label_lab, (512, 512))

        # Convert to PIL Image for transformations
        image_lab = Image.fromarray(image_lab)
        label_lab = Image.fromarray(label_lab)

        # max and min value of image_lab
        # print("image_lab max: ", np.max(image_lab))
        # print("image_lab min: ", np.min(image_lab))

        image_lab, label_lab = self.hflip(image_lab, label_lab)

        # label_lab = np.array(label_lab, dtype='float32') / 255.0
        label_lab = np.array(label_lab, dtype='float32')

        image_nom = self.trans(image_lab)
        # print("image_nom max: ", image_nom.max())
        # print("image_nom min: ", image_nom.min())
        label_lab = np.array([label_lab])
        # print("image_nom shape: ", image_nom.shape)
        # label_lab shape:  (1, 512, 512, 3)
        # image_nom shape:  torch.Size([3, 512, 512])
        # align the shape of label_lab to image_nom
        label_lab = label_lab.transpose(3, 0, 1, 2)
        # label_lab shape:  (3, 1, 512, 512)
        # align the shape of label_lab to image_nom
        label_lab = np.squeeze(label_lab)
        # print("label_lab shape: ", label_lab.shape)

        image_ori =  np.array(image_lab, dtype='float32').transpose(2, 0, 1)
        sample = {'O': image_nom, 'B': label_lab, 'image': np.array(image_lab, dtype='float32').transpose(2, 0, 1) / 255, "image_ori": np.array(image_lab, dtype='float32').transpose(2, 0, 1)}

        return sample


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.root = '/home/zhxing/Datasets/SRD_inpaint4shadow_fix/'
        # self.root = '/home/zhxing/Datasets/ISTD+/'
        # self.root = '/home/zhxing/Datasets/DESOBA_xvision/'

        self.imgs = open(self.root + 'test_dsc.txt').readlines()
        self.file_num = len(self.imgs)

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])

    def __len__(self):
        return self.file_num 

    def __getitem__(self, index):
        image_path, label_path = self.imgs[index % self.file_num][:-1].split(' ')
        image = cv2.imread(self.root + image_path)
        label = cv2.imread(self.root + label_path)

        # Convert to LAB color space
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        label_lab = cv2.cvtColor(label, cv2.COLOR_BGR2LAB)

        # Resize images
        image_lab = cv2.resize(image_lab, (512, 512))
        label_lab = cv2.resize(label_lab, (512, 512))

        # Convert to PIL Image for transformations
        image_lab = Image.fromarray(image_lab)

        label_lab = np.array(label_lab, dtype='float32') / 255.0
        image_nom = self.trans(image_lab)
        
        label_lab = np.array([label_lab])
        # print("image_nom shape: ", image_nom.shape)
        # label_lab shape:  (1, 512, 512, 3)
        # image_nom shape:  torch.Size([3, 512, 512])
        # align the shape of label_lab to image_nom
        label_lab = label_lab.transpose(3, 0, 1, 2)
        # label_lab shape:  (3, 1, 512, 512)
        # align the shape of label_lab to image_nom
        label_lab = np.squeeze(label_lab)
        # print("label_lab shape: ", label_lab.shape)

        # print the range of image_nom
        # print("image_nom max: ", image_nom.max())

        image_ori =  np.array(image_lab, dtype='float32').transpose(2, 0, 1)

        sample = {'O': image_nom, 'B': label_lab, 'image': np.array(image_lab), "image_ori": image_ori}


        return sample
