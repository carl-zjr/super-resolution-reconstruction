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
from train import setup

def down_and_up_sampling(image, save_name, upsampling):
    
    opt = setup()
    # create output folder
    try:
        os.makedirs('output/high_res_fake')
        os.makedirs('output/high_res_real')
        os.makedirs('output/low_res')
    except OSError:
        pass

    if torch.cuda.is_available() and not opt.cuda:
        print('[WARNING]: You have a CUDA device, so you should probably run with --cuda')

    transform = transforms.Compose([transforms.RandomCrop((
                                                                image.size[0],
                                                                image.size[1])),
                                    transforms.Pad(padding = 0),
                                    transforms.ToTensor()])
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    # [down sampling] down-sampling part
    scale = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((int(image.size[1] / opt.upSampling), int(image.size[0] / opt.upSampling))),
                                transforms.Pad(padding=0),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                                        mean = [0.485, 0.456, 0.406],
                                                        std = [0.229, 0.224, 0.225])])
    
    # Equivalent to un-normalizing ImageNet (for correct visualization)
    unnormalize = transforms.Normalize(
                                            mean = [-2.118, -2.036, -1.804],
                                            std = [4.367, 4.464, 4.444])

    if opt.dataset == 'folder':
        # folder dataset
        dataset = datasets.ImageFolder(root = opt.dataroot, transform = transform)
    elif opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root = opt.dataroot, download = True, train = False, transform = transform)
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root = opt.dataroot, download = True, train = False, transform = transform)
    assert dataset

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size = opt.batchSize,
                                             shuffle = False,
                                             num_workers = int(opt.workers))

    my_loader = transforms.Compose([transforms.ToTensor()])
    image = my_loader(image)

    # [paras] loading paras from .pth files
    generator = Generator(16, opt.upSampling)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))

    # For the content loss
    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained = True))

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

    # print('Test started...')
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    # Set evaluation mode (not training)
    generator.eval()
    discriminator.eval()

    data = image
    for i in range(1):
        # Generate data
        high_res_real = data
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
            high_res_fake = generator(Variable(low_res)) # >>> create hr images

        save_image(unnormalize(high_res_real[0]), 'output/high_res_real/' + save_name)
        save_image(unnormalize(high_res_fake[0]), 'output/high_res_fake/' + save_name)
        save_image(unnormalize(low_res[0]), 'output/low_res/' + save_name)

def padding(imageA, imageB, factor, savename):
    modeA = Image.open(imageA)
    lengthA, widthA = modeA.size
    
    modeB = Image.open(imageB)
    lengthB, widthB = modeB.size
    # padding = 0, fill=255, padding_mode='constant'
    transform = transforms.Compose([
                                        transforms.Pad(padding=0),
                                        transforms.CenterCrop((widthA * factor, lengthA * factor)),
                                        transforms.ToTensor()])
    modeB = transform(modeB)
    save_image(modeB, savename)

def create_test_data(path, count, bar):
    count = 0
    size = len(os.listdir(path))
    for img in os.listdir(path):
        '''print('>>> process image : {}'.format(img))'''
        image = Image.open(path + img)
        down_and_up_sampling(image, save_name = img, upsampling = 4)
        count += 100 / size
        bar.setValue(count)
    
    try:
        os.makedirs(os.getcwd() + r'\output\padding_fake')
    except OSError:
        pass
    
    for f in os.listdir(os.getcwd() + r'\output\high_res_fake'):
        padding(os.getcwd() + r'\output\high_res_real\\' + f,
                os.getcwd() + r'\output\high_res_fake\\'  + f,
                1, os.getcwd() + r'\output\padding_fake\\'  + f)
    lr_path  = os.getcwd() + r'\output\low_res'
    hr_real_path = os.getcwd() + r'\output\high_res_real'
    hr_fake_path = os.getcwd() + r'\output\high_res_fake'
    return lr_path, hr_real_path, hr_fake_path


   
