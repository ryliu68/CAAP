# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T
import socket

class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c,h,w = tensor.size()
   
        if c != 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h*w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std() 
        t = t.sub_(m).div_(s+1e-6)
        tensor[idx] = t
        
        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats = self.outchannels, dim = 0)
    
        return tensor


class PalmPrint(data.Dataset):
    '''
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None 
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    '''
    
    def __init__(self, data_dir,norm, transforms=None, train=True, co3net=False,imside = 128, outchannels = 1):        

        self.train = train
        self.norm = norm

        self.co3net = co3net

        self.imside = imside # 128, 224
        self.chs = outchannels # 1, 3

        self.data_dir = data_dir        

        self.transforms = transforms

        if transforms is None:

            if self.norm =="aug_0.5":
                if not train: 
                    self.transforms = T.Compose([        
                        # T.Resize(self.imside),                  
                        T.ToTensor(),   
                        T.Normalize([0.5],[0.5]),  
                        ]) 
                else:
                    self.transforms = T.Compose([ 
                        # T.Resize(self.imside),                  

                        T.ToTensor(),
                        T.Normalize([0.5],[0.5]),
                        ])
                    
            elif "aug_org" in self.norm:
                if not train: 
                    self.transforms = T.Compose([        
                        # T.Resize(self.imside),                  
                        T.ToTensor(),   
                        NormSingleROI(outchannels=self.chs)     
                        ]) 
                else:
                    self.transforms = T.Compose([ 
                        # T.Resize(self.imside),
                        T.RandomChoice(transforms=[
                            T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),# 0.3 0.35
                            T.RandomResizedCrop(size=self.imside, scale=(0.8,1.0), ratio=(1.0, 1.0)),
                            T.RandomPerspective(distortion_scale=0.15, p=1),# (0.1, 0.2) (0.05, 0.05)
                            T.RandomChoice(transforms=[
                                T.RandomRotation(degrees=10,  expand=False, center=(0.5*self.imside, 0.0)),
                                T.RandomRotation(degrees=10, expand=False, center=(0.0, 0.5*self.imside)),
                            ]),
                        ]),     

                        T.ToTensor(),
                        NormSingleROI(outchannels=self.chs)                   

                        ])
            elif self.norm == "none":
                self.transforms = T.Compose([
                    T.ToTensor(),
                ])
            else:
                raise NameError
                
        self._read_data()

 
    def _read_data(self):
        """Load dataset file and normalize structure across formats.
        Supports both dict-with-subdict (images/labels) and list-of-pairs formats.
        """
        raw = torch.load(self.data_dir, weights_only=False)

        def to_img_lbl(split):
            # split could be dict {'images':..., 'labels':...} or list[(img,lbl),...]
            if isinstance(split, dict) and 'images' in split and 'labels' in split:
                return split['images'], split['labels']
            if isinstance(split, (list, tuple)) and len(split) > 0:
                # list of (image, label)
                imgs = [it[0] for it in split]
                lbls = [int(it[1]) if not isinstance(it[1], (int, torch.Tensor)) else int(it[1]) for it in split]
                return imgs, lbls
            raise TypeError('Unsupported dataset split format: {}'.format(type(split)))

        if self.train:
            images, labels = to_img_lbl(raw.get('train', raw.get('Train', {})))
        else:
            images, labels = to_img_lbl(raw.get('test', raw.get('Test', {})))

        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index]

       

        data = self.images[index] 
        data = self.transforms(data)    

        if self.co3net:
            idx2 = np.random.choice(np.arange(len(self.labels))[np.array(self.labels) == label])

            if self.train == True:
                while(idx2 == index):
                    idx2 = np.random.choice(np.arange(len(self.labels))[np.array(self.labels) == label])
            else:
                idx2 = index

            data2 = self.images[idx2] 
            data2 = self.transforms(data2)

            data = [data,data2]

        return data, int(label)#, img_path
    

    def __len__(self):
        return len(self.labels)


def get_data(args):
    hostname = socket.gethostname()
    if hostname =="PC":
        data_path="/home/mrliu/work/datasets/Palmprint/data_pth"
    else:
        data_path="/home/renyang/jiale/Palmprint_Attack_UAP/data_pth"

    if args.dataset =="tongji":
        args.data_dir = F'{data_path}/tongji_org_128.pth'
    elif args.dataset == "iitd":
        args.data_dir = F'{data_path}/IITD_org_128.pth'
    elif args.dataset in ("aisec", "ynu"):
        args.data_dir = F'{data_path}/AISEC_org_128.pth'
    else:
        raise NameError

    trainset = PalmPrint(data_dir=args.data_dir, norm=args.norm, transforms=None, train=True, co3net=args.co3net)
    testset = PalmPrint(data_dir=args.data_dir, norm=args.norm, transforms=None, train=False, co3net=args.co3net)


    num_workers = getattr(args, "num_workers", 4)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader, testloader
