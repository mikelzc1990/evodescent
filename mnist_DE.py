import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
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

parser = argparse.ArgumentParser(description='PyTorch DE Deep CNN weights training')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
parser.add_argument('--pop_size', type=int, default=20, help='population size')
parser.add_argument('--tour_size', type=int, default=4, help='tournament size')
parser.add_argument('--p_crx', type=float, default=0.9, help='probability for crossover')
parser.add_argument('--scale_factor', type=float, default=0.75, help='scaling factor F for DE')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--save', type=str, default='DE', help='experiment name')
parser.add_argument('--n_batch', type=int, default=5, help='num of batches used to calculate accuracy')
parser.add_argument('--gens', type=int, default=300, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--min_learning_rate', type=float, default=0.0, help='minimum learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
args = parser.parse_args()

args.save = 'MNIST-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

# prepare hyper-parameters
device = 'cuda'
# device = 'cpu'
seed = args.seed
population_size = args.pop_size
tournament_size = args.tour_size
p_crx = args.p_crx
scale_factor = args.scale_factor
batch_size = args.batch_size
# calculate number of batches for each individual
n_batch = args.n_batch
# calculate report frequency
report_freq = population_size*3
generations = args.gens

assert n_batch > 0

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


class Net(nn.Module):
    def __init__(self, init_channels=args.init_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=init_channels,
                               kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=init_channels,
                               out_channels=2 * init_channels,
                               kernel_size=3, padding=1, bias=False)
        self.classifier = nn.Linear(in_features=2 * init_channels, out_features=10)

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

        return self.classifier(x)


def random_weights(dim, mu=0., sigma=1.):
    # gaussian
    return np.random.normal(mu, sigma, dim)


def random_combination(iterable, sample_size):
    """Random selection from itertools.combinations(iterable, r)."""
    pool = tuple(iterable)
    n = len(pool)

    indices = sorted(random.sample(range(n), sample_size))

    return tuple(pool[i] for i in indices)


# create a model with specified weights
def create_model(individual=None):
    model = Net()
    # we use sgd to learn the classification layer
    # turn off gradients for all other layers
    for name, param in model.named_parameters():
        if not ('classifier' in name):
            param.requires_grad = False

    if individual is not None:
        # loading parameters
        # first - we load the parameters optimized by EA
        params_to_load = individual[1]['x']
        lb = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                ub = lb + param.nelement()
                layer_size = tuple(param.size())
                param.data = (torch.from_numpy(params_to_load[lb:ub].reshape(
                    layer_size))).type(torch.FloatTensor)
                lb += param.nelement()
        assert ub == len(params_to_load)

        # second - we load classifier's weights and bias
        if individual[1]['classifier_weights'] is not None:
            model.classifier.weight.data = individual[1]['classifier_weights']
            model.classifier.bias.data = individual[1]['classifier_bias']

    return model.to(device)


# def load_ea_parameters(model, params_to_load):
#     lb = 0
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             ub = lb + param.nelement()
#             layer_size = tuple(param.size())
#             param.data = (torch.from_numpy(params_to_load[lb:ub].reshape(
#                 layer_size))).type(torch.FloatTensor).to(device)
#             lb += param.nelement()
#     assert ub == len(params_to_load)
#     return model


def uniform_crossover(p, q, prob_crx):
    c = np.full(p.shape, np.nan)

    for i in range(p.shape[0]):
        if np.random.rand() <= prob_crx:
            c[i] = p[i]
        else:
            c[i] = q[i]

    return c


def differential_recombination(parents, F, prob_crx):
    # child = parent1 + F * (parent2 - parent3)
    # child crossover parent4
    assert len(parents) > 3

    order = (np.random.permutation(len(parents))).astype(int).tolist()

    p1 = parents[order[0]]
    p2 = parents[order[1]]
    p3 = parents[order[2]]
    p4 = parents[order[3]]

    # for NSDE - DE w/ neighborhood search
    if np.random.rand() < 0.5:
        F = np.random.normal(0.5, 0.5, 1)
    else:
        F = np.random.standard_cauchy(1)

    c = uniform_crossover(p1[1]['x'] + F * (p2[1]['x'] - p3[1]['x']), p4[1]['x'], prob_crx)

    # bounce back if you want weights to be between bounds
    c[c < -1] = -5
    c[c > 1] = 5
    return c, p4


def evaluate(pop, train_queue, criterion):
    # every individual in population will only be evaluated on a few mini-batch
    pop_eval = []

    for p in pop:
        pop_eval.append(train(p, train_queue, criterion))

    return pop_eval


# Training
def train(indv, train_queue, criterion):

    net = create_model(indv)

    # extract parameter for classifier:
    for name, param in net.named_parameters():
        if not ('classifier' in name):
            param.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    optimizer = optim.SGD(parameters,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_batch, eta_min=args.min_learning_rate)

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        # only update for pre-defined # of epochs
        if step >= n_batch:
            break

        # scheduler.step()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # record back-propagation optimized weights
    indv_evaled = (100.*correct/total, {
        'x': indv[1]['x'],
        'classifier_weights': net.classifier.weight.data.cpu(),
        'classifier_bias': net.classifier.bias.data.cpu()
    })

    return indv_evaled


def infer(elite, valid_queue, criterion):
    # net = Net().to(device).eval()
    net = create_model(elite)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % 50 == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    logging.info('valid acc %f', 100. * correct / total)

    return test_loss/total, acc


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
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # just for getting the number of trainable parameters
    model = create_model()
    n_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logging.info("param size = %d", n_params)
    del model

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # initialization
    population = []
    for _ in range(population_size):
        weights = random_weights(n_params, sigma=0.01)
        # dict = {'x': parameters for ea to optimize,
        #         'classifier_weights': classifier's weights optimized by back-propagation
        #         'classifier_bias'   : classifier's bias optimized by back-propagation
        # }
        # population = [(acc1, dict1), (acc2, dict2), ..., (accN, dictN)]
        population.append((0, {
            'x': weights,
            'classifier_weights': None,
            'classifier_bias': None,
        }))

    # evaluation
    population = evaluate(population, train_loader, criterion)
    elite_idx = np.argmax([x[0] for x in population])  # find individual w/ highest accuracy
    logging.info('train acc %04d %f', 0, population[elite_idx][0])

    n_child_survived = 0
    # main loop of evolution
    for gen in range(1, generations + 1):
        sample = random_combination(population, tournament_size)
        new_weights, parent_crx = differential_recombination(sample, scale_factor, p_crx)

        child = [(0, {
            'x': new_weights,
            'classifier_weights': None,
            'classifier_bias': None,
        })]
        child = evaluate(child, train_loader, criterion)

        if child[0][0] >= parent_crx[0]:
            # replace the crossover parent with child if child has better fitness
            # replace loser in population with child
            remove_idx = [i for i in range(len(population))
                          if np.all(population[i][1]['x'] == parent_crx[1]['x'])][0]
            population.pop(remove_idx)
            population += child
            n_child_survived += 1

        if gen % report_freq == 0:
            elite_idx = np.argmax([x[0] for x in population])
            logging.info('train acc %04d %.2f %f', gen, 100*n_child_survived/report_freq, population[elite_idx][0])
            n_child_survived = 0

    elite_idx = np.argmax([x[0] for x in population])
    infer(population[elite_idx], test_loader, criterion)


if __name__ == '__main__':
    main()
    # net = Net()
    # weight = net.classifier.weight.data.cpu().numpy()
    # weight_new = weight[:]
    # weight_new = weight_new.flatten()
    # weight_new = weight_new.reshape(weight.shape)
    # print(weight.shape)
    # print(weight_new.shape)
    # print(weight_new - weight)
