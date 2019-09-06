# -*- coding:utf-8 -*-

import os
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image

def read_image_size(path):
    # read image from path
    image = cv2.imread(path)
    mode = Image.open(path)
    
    # shape of image
    width, height, channels = image.shape
    # width, height = mode.size
    
    # value type of image and r/g/b channel
    t = image.dtype
    b = image[:, :, 0]
    g = image[:, :, 0]
    r = image[:, :, 0]

    return width, height, mode.mode, mode.format

def images_check(data_dir):
    height, width = [], []
    mode, fmat = set(), set()
    for f in os.listdir(data_dir):
        w, h, m, f = read_image_size(data_dir + f)
        width.append(w)
        height.append(h)
        mode.add(m)
        fmat.add(f)

    log = ''
    # display images set
    log += '-'*32 + '\n'
    log += '{:<20}:{:^10}'.format('image mode', str(mode)) + '\n'
    log += '{:<20}:{:^10}'.format('image format', str(fmat)) + '\n'
    log += '{:<20}:{:^10}'.format('size of data set', len(height)) + '\n'
    log += '{:<20}:{:^10}'.format('average width', sum(width) / len(width)) + '\n'
    log += '{:<20}:{:^10}'.format('average hight', sum(height) / len(height)) + '\n'
    log += '{:<20}:{:^10}'.format('max width', max(width)) + '\n'
    log += '{:<20}:{:^10}'.format('max height', max(height)) + '\n'
    log += '{:<20}:{:^10}'.format('min width', min(width)) + '\n'
    log += '{:<20}:{:^10}'.format('min height', min(height)) + '\n'
    log += '-'*32 + '\n'

    return log


    
    

