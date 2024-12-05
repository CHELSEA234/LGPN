# coding: utf-8

import torch
import torchvision
from torch.utils import data
import h5py
import os
import sys
from glob import glob
import random
import numpy as np
import json
from PIL import Image
import numpy as np
import imageio.v2 as imageio
import cv2
from ground_truth import label_dict, set_1, set_2, set_3, set_4

# Image transformation
def get_image_transformation(use_laplacian=False, normalize=True):
    transforms = []
    transforms.extend(
                    [torchvision.transforms.ToPILImage(), # Next line takes PIL images as input (ToPILImage() preserves the values in the input array or tensor)
                     torchvision.transforms.Resize((256,256)),
                     torchvision.transforms.ToTensor(), # To bring the pixel values in the range [0,1]
                     torchvision.transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))]
                    )
    return torchvision.transforms.Compose(transforms)

# Real Image Samples
def return_real_image_list():

    transform = get_image_transformation(use_laplacian=False)

    ImageNet_path = "/user/guoxia11/cvlshare/cvl-guoxia11/RED_LIVE/ImageNet/"
    ImageNet_dir  = os.path.join(ImageNet_path, 'ImageNet_dir')
    ImageNet_lst  = os.path.join(ImageNet_path, 'ImageNet_list.txt')
    # ImageNet_lst  = ImageNet_lst

    MNIST_path = "/user/guoxia11/cvlshare/cvl-guoxia11/RED_LIVE/MNIST_data/"
    MNIST_dir  = os.path.join(MNIST_path, 'MNIST_data_train')
    MNIST_lst  = os.path.join(MNIST_path, 'MNIST_train_list.txt')
    # MNIST_lst  = MNIST_lst[::2]

    cifar_path = "/user/guoxia11/cvlshare/cvl-guoxia11/RED_LIVE/cifar10/"
    cifar_dir  = os.path.join(cifar_path, 'cifar_train')
    cifar_lst  = os.path.join(cifar_path, 'cifar_10_train_list.txt')
    # cifar_lst  = cifar_lst[::2]

    FFHQ_path = "/user/guoxia11/cvlshare/cvl-guoxia11/IMDL/REAL/"
    FFHQ_dir  = os.path.join(FFHQ_path, 'FFHQ')
    FFHQ_lst  = os.path.join(FFHQ_path, 'FFHQ.txt')

    CelebAHQ_path = "/user/guoxia11/cvlshare/cvl-guoxia11/IMDL/REAL/"
    CelebAHQ_dir  = os.path.join(CelebAHQ_path, 'CelebAHQ')
    CelebAHQ_lst  = os.path.join(CelebAHQ_path, 'CelebAHQ.txt')

    LSUN_path = "/user/guoxia11/cvlshare/cvl-guoxia11/IMDL/REAL/"
    LSUN_dir  = os.path.join(LSUN_path, 'LSUN')
    LSUN_lst  = os.path.join(LSUN_path, 'LSUN.txt')
    # LSUN_lst  = LSUN_lst[::5]

    # import sys;sys.exit(0)
    def check_image(dir_input, list_input, integer=2):
        '''combining the list.'''
        cur_real_img_lst = []
        f = open(list_input, 'r')
        lines = f.readlines()
        lines.sort()
        lines = lines[::integer]
        for line in lines:
            line = line.strip()
            fname = os.path.join(dir_input, line)
            if os.path.isfile(fname) == False:
                print(fname)
            else:
                cur_real_img_lst.append(fname)
        return cur_real_img_lst

    real_img_list = []
    real_img_list += check_image(ImageNet_dir, ImageNet_lst,6)
    real_img_list += check_image(MNIST_dir, MNIST_lst, 5)
    real_img_list += check_image(cifar_dir, cifar_lst, 5)
    real_img_list += check_image(FFHQ_dir, FFHQ_lst,3)
    real_img_list += check_image(CelebAHQ_dir, CelebAHQ_lst,3)
    real_img_list += check_image(LSUN_dir, LSUN_lst,10)
    real_img_list.sort()
    return real_img_list

# Main dataloader 
def get_dataloader(
                    img_path,
                    train_dataset_names,
                    label_list,
                    folder_lst,
                    manipulations_dict,
                    normalize=True,
                    mode='train',
                    bs=32,
                    workers=4,
                    cv=1
                    ):
    transform = get_image_transformation(use_laplacian=False, normalize=normalize)
    params = {'batch_size': bs,
              'shuffle': (mode=='train'),
              'num_workers': workers,
              'drop_last' : (mode=='train')
            }
    data_dict = {}
    cls0_sam_list, cls1_sam_list = [], []
    dataset_cls0_dict = {}
    dataset_cls1_dict = {}

    set_1_count, set_2_count, set_3_count, set_4_count = 0,0,0,0
    for idx, folder_name in enumerate(folder_lst):
        cur_lst = glob(f'{img_path}/{folder_name}/*')
        cur_lst.sort()
        folder_name_ = folder_name.split('/')[-1]
        
        if cv == 1:
            set_train = set_2 + set_3 + set_4
            set_val = set_1
        elif cv == 2:
            set_train = set_1 + set_3 + set_4
            set_val = set_2
        elif cv == 3:
            set_train = set_1 + set_2 + set_4
            set_val = set_3
        elif cv == 4: 
            set_train = set_1 + set_2 + set_3
            set_val = set_4
        else:
            raise ValueError

        if mode == 'train': ## GX: put folders not in set_val in training samples. 
            if folder_name_ not in set_val:
                cls0_sam_list += cur_lst
        elif mode != 'train' and folder_name_ in set_val:   ## GX: put folders in set_val as val samples.
            cls0_sam_list += cur_lst
    print(f"{cv}th cross validation with {mode}, the image number is: {len(cls0_sam_list)}.")
    cls0_sam_list.sort()
    
    redDataset = RedDataset(cls0_sam_list, 
                            manipulations_dict, 
                            transform=transform)
    if mode == 'train':
        real_img_list = return_real_image_list()
        realDataset = RealsampleDataset(img_list=real_img_list, transform=transform)
        redDataset  = data.ConcatDataset([redDataset, realDataset])
    else:
        pass
    redGenerator = data.DataLoader(redDataset,**params)
    return redGenerator, redDataset

class RealsampleDataset(data.Dataset):
    def __init__(self, img_list, transform):
        super(RealsampleDataset, self).__init__()
        self.img_list = img_list
        self.transform = transform
        self.label0 = np.array([0] * 55)  # dummy red label
        # self.label1 = np.array([0,1])
        self.label1 = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_name = random.choice(self.img_list)
        image      = imageio.imread(image_name, pilmode='RGB')
        # image = cv2.imread(image_name)
        rgb_preproc = self.transform(image)
        return rgb_preproc, self.label0, self.label1, image_name

class RedDataset(data.Dataset):
    def __init__(self, img_list, manipulations_dict, transform):
        super(RedDataset, self).__init__()
        self.img_list = img_list
        self.dname_to_id = manipulations_dict
        self.transform = transform
        self.label1 = 1

    def _decompose(self, image_name, filter_list=[3,5,7]):
        im_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        hifreq_list = []
        for filter_size in filter_list:
            blur = cv2.GaussianBlur(im_gray,(filter_size,filter_size),1)
            hifr = im_gray - blur
            hifr = hifr[:,:,np.newaxis]
            hifreq_list.append(hifr)
        hifr_input = np.concatenate(hifreq_list, axis=-1)
        return hifr_input

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_name = random.choice(self.img_list)
        image    = imageio.imread(image_name, pilmode='RGB')
        GM_name  = image_name.split('/')[-2]
        try:
            self.label = np.array(label_dict[GM_name])
        except:
            print(f"Cannot find {GM_name}")
        if image.shape[-1] == 4:
            assert ValueError, print(f"The image is not the RGB format.")
        rgb_preproc = self.transform(image)
        return rgb_preproc, self.label, self.label1, image_name