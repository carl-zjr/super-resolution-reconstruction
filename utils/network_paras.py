# -*- coding:utf-8 -*-

import torch
import torchvision
from torchsummary import summary
from models import Generator, Discriminator, FeatureExtractor
from train import setup

def printer(string):
    main_string = '#' + ' ' * 30 + string + ' ' * 30 + '#'
    print('#' * len(main_string))
    print('#' + ' ' * len(' ' * 30 + string + ' ' * 30) + '#' )
    print(main_string)
    print('#' + ' ' * len(' ' * 30 + string + ' ' * 30) + '#' )
    print('#' * len(main_string))

def print_network():
    opt = setup()
    generator = Generator(16, opt.upSampling)
    if opt.generatorWeights != '':
        generator.load_state_dict(torch.load(opt.generatorWeights))

    discriminator = Discriminator()
    if opt.discriminatorWeights != '':
        discriminator.load_state_dict(torch.load(opt.discriminatorWeights))

    feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained = True))

    printer('generator')
    summary(generator.cuda(), (3, 32, 32))
    printer('discriminator')
    summary(discriminator.cuda(), (3, 32, 32))
    printer('feature_extractor')
    summary(feature_extractor.cuda(), (3, 32, 32))

