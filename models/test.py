#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.backdoor_data import MyDataset_cifar, MyDataset_mnist, MyDataset_cifar_DBA, MyDataset_mnist_DBA
from utils.options import args_parser

args = args_parser()
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(device), target.to(device)
        log_probs, _ = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def test_img_poison(net_g, datatest, args, epoch):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    datatest = list(datatest)
    if args.dataset == 'cifar':
        if args.attack_methods == 'CBA':
            datatest_poison = MyDataset_cifar(datatest[:3000], args.target_label, portion=1.0,
                                             mode="test")
        elif args.attack_methods == 'DBA':
            datatest_poison = MyDataset_cifar_DBA(args, datatest[:3000], args.target_label,
                                                 portion=1.0, mode="test", idx=-1)
    elif args.dataset == 'mnist':
        if args.attack_methods == 'CBA':
            datatest_poison = MyDataset_mnist(datatest[:3000], args.target_label, portion=1.0,
                                             mode="test")
        elif args.attack_methods == 'DBA':
            datatest_poison = MyDataset_mnist_DBA(args, datatest[:3000], args.target_label,
                                                 portion=1.0, mode="test", idx=-1)
    data_loader = DataLoader(datatest_poison, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(device), target.to(device)
        log_probs, _ = net_g(data)
        # sum up batch loss
        # target = torch.argmax(target, dim=1)
        test_loss += F.cross_entropy(log_probs, target.long(), reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss