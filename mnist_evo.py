import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
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


parser = argparse.ArgumentParser(description='PyTorch Evolutionary Deep CNN weights training')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--pop_size', type=int, default=20, help='population size')
parser.add_argument('--gens', type=int, default=1000, help='maximum number of generations')
parser.add_argument('--tour_size', type=int, default=10, help='tournament size')
parser.add_argument('--p_mut', type=float, default=0.1, help='probability for mutation')
parser.add_argument('--eta_m', type=float, default=30.0, help='polynomial mutation parameter')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')

args = parser.parse_args()

args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

# prepare hyper-parameters
device = 'cuda'
seed = args.seed
population_size = args.pop_size
generations = args.gens
tournament_size = args.tour_size
p_mut = args.p_mut
eta_m = args.eta_m
batch_size = args.batch_size
# calculate number of batches for each individual
n_batch = int(50000 / population_size / batch_size)
assert n_batch > 0

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


class Net(nn.Module):
    def __init__(self, init_channels=8):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=init_channels,
                               kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=init_channels,
                               out_channels=2 * init_channels,
                               kernel_size=3, padding=1, bias=False)
        self.fc = nn.Linear(in_features=2 * init_channels, out_features=10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)

        return self.fc(x)


def random_weights(dim, mu=0., sigma=1.):
    # gaussian
    return np.random.normal(mu, sigma, dim)


def load_parameters(model, params_to_load):
    lb = 0
    for name, param in model.named_parameters():
        ub = lb + param.nelement()
        layer_size = tuple(param.size())
        param.data = (torch.from_numpy(params_to_load[lb:ub].reshape(
            layer_size))).type(torch.FloatTensor).to(device)
        lb += param.nelement()
    assert ub == len(params_to_load)
    return


def evaluate(pop, train_queue, criterion):
    # every individual in population will only be evaluated on one mini-batch
    # assuming mini-batch size = total_num_training_data / population size
    pop_eval, indv_counter, batch_counter = [], 0, 0
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(train_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            # when we move on to next individual in population
            if batch_counter == 0:
                model = Net().to(device).eval()
                load_parameters(model, pop[indv_counter][1])
                correct = 0
                total = 0
            outputs = model(inputs)
            loss = criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            batch_counter += 1
            if batch_counter > n_batch:
                # reset counter
                batch_counter = 0
                indv_counter += 1
                # record performance
                acc = 100. * correct / total
                pop_eval.append((acc, pop[indv_counter][1]))

            if not (len(pop_eval) < len(pop)):
                break

    return pop_eval


def main():
    # ------------- main routine ------------------ #
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", args)

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False)

    # just for getting the number of trainable parameters
    model = Net()
    n_params = utils.count_parameters_in_MB(model)
    logging.info("param size = %fMB", n_params)
    del model

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # initialization
    population = []
    for _ in range(population_size):
        weights = random_weights(n_params, sigma=0.01)
        # population = [(acc1, weights1), (acc2, weights2), ..., (accN, weightsN)]
        population.append((0, weights))

    # evaluation
    population = evaluate(population, train_loader, criterion)
    print(population)
    1/0
    elite_idx = np.argmin([x[0] for x in population])
    print('train %03d %e %e %f' % (0, p_mut,
                                   population[elite_idx][0],
                                   population[elite_idx][-1]))

    # main loop of evolution
    for gen in range(1, generations + 1):
        sample = random_combination(population, tournament_size)
        # best from the sample becomes parent
        tmp = sorted(sample, key=lambda i: i[0])

        winner, loser = tmp[0], tmp[-1]

        # update mutation probability - linear drop
        p_mut = p_mut - (p_mut - 1 / n_params_trainable) / generations
        child = [(np.inf, PolynomialMutation(eta_mut=eta_m,
                                             prob_mut=p_mut,
                                             xl=-2, xu=2).do(winner[1]))]
        child = evaluate(child, train_loader, criterion)

        # replace loser in population with child
        remove_idx = [i for i in range(len(population))
                      if np.all(population[i][1] == loser[1])][0]

        population.pop(remove_idx)
        population += child
        if gen % 100 == 0:
            elite_idx = np.argmin([x[0] for x in population])
            print('train %03d %e %e %f' % (0, p_mut,
                                           population[elite_idx][0],
                                           population[elite_idx][-1]))

