# -*- coding:utf-8 -*-

import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import Generator, Discriminator, FeatureExtractor
from PIL import Image
import numpy as np
import models
from train import setup

def upsampling(path, picture_name, upsampling):
    opt = setup()
    # image = Image.open(os.getcwd() + r'\images\\' + path)
    image = Image.open(path)
    opt.imageSize = (image.size[1], image.size[0])

    log = '>>> process image : {} size : ({}, {}) sr_reconstruct size : ({}, {})'.format(picture_name,
                                                                                         image.size[0],
                                                                                         image.size[1],
                                                                                         image.size[0] * upsampling,
                                                                                         image.size[1] * upsampling)        
    try:
        os.makedirs(os.getcwd() + r'\output\result')
    except OSError:
        pass

    if torch.cuda.is_available() and not opt.cuda:
        print('[WARNING] : You have a CUDA device, so you should probably run with --cuda')

    transform = transforms.Compose([transforms.RandomCrop(opt.imageSize),
                                    transforms.Pad(padding=0),
                                    transforms.ToTensor()])
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
    # Equivalent to un-normalizing ImageNet (for correct visualization)
    unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804],
                                       std = [4.367, 4.464, 4.444])
    scale = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(opt.imageSize),
                                transforms.Pad(padding=0),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                     std = [0.229, 0.224, 0.225])
                                ])

    if opt.dataset == 'folder':
        # folder dataset
        dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
    elif opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.dataroot, download=True, train=False, transform=transform)
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root=opt.dataroot, download=True, train=False, transform=transform)
    assert dataset
    
    dataloader = transforms.Compose([transforms.ToTensor()])
    image = dataloader(image)

    # loading paras from networks
    generator = Generator(16, opt.upSampling)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))

    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))

    content_criterion = nn.MSELoss()
    adversarial_criterion = nn.BCELoss()

    target_real = Variable(torch.ones(opt.batchSize, 1))
    target_fake = Variable(torch.zeros(opt.batchSize, 1))

    # if gpu is to be used
    if opt.cuda:
        generator.cuda()
        discriminator.cuda()
        feature_extractor.cuda()
        content_criterion.cuda()
        adversarial_criterion.cuda()
        target_real = target_real.cuda()
        target_fake = target_fake.cuda()

    low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize[0], opt.imageSize[1])

    # Set evaluation mode (not training)
    generator.eval()
    discriminator.eval()

    # Generate data
    high_res_real = image

    # Downsample images to low resolution
    low_res = scale(high_res_real)
    low_res = torch.tensor([np.array(low_res)])

    high_res_real = normalize(high_res_real)
    high_res_real = torch.tensor([np.array(high_res_real)])
            
    # Generate real and fake inputs
    if opt.cuda:
        high_res_real = Variable(high_res_real.cuda())
        high_res_fake = generator(Variable(low_res).cuda())
    else:
        high_res_real = Variable(high_res_real)
        high_res_fake = generator(Variable(low_res))

    save_image(unnormalize(high_res_fake[0]), './output/result/' + picture_name)
    return log

def super_resolution_reconstruction(path):
    for i, image_name in enumerate(path):
        upsampling(image_name, picture_name = image_name, upsampling = 4)
