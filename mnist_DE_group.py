import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import os
import sys
import time
import math
import random
import logging
import argparse
import numpy as np

import utils


parser = argparse.ArgumentParser(description='PyTorch Grouping DE Deep CNN weights training')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--pop_size', type=int, default=20, help='population size')
parser.add_argument('--tour_size', type=int, default=4, help='tournament size')
parser.add_argument('--p_crx', type=float, default=0.9, help='probability for crossover')
parser.add_argument('--scale_factor', type=float, default=0.75, help='scaling factor F for DE')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--save', type=str, default='DE-Group', help='experiment name')
parser.add_argument('--cycles', type=int, default=10, help='num of cycles for group shuffling')
parser.add_argument('--gens', type=int, default=10, help='num of generations for each ')
args = parser.parse_args()

args.save = 'MNIST-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

# prepare hyper-parameters
device = 'cuda'
seed = args.seed
population_size = args.pop_size
tournament_size = args.tour_size
p_crx = args.p_crx
scale_factor = args.scale_factor
batch_size = args.batch_size
# calculate number of batches for each individual
n_batch = int(50000 / population_size / batch_size)
# calculate report frequency
report_freq = population_size*3
generations = population_size*3*args.epochs

assert n_batch > 0