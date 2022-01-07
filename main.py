#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import h5py
import os
from utils.sampling import mnist_iid, cifar_iid, build_classes_dict, sample_dirichlet_train_data
from utils.options import args_parser
from models.update import LocalUpdate
from models.nets import LeNet, MLP, ResNet18, ResNet50
from models.Fed_aggregation import aggregation
from models.test import test_img, test_img_poison
import random
from random import choice


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.use_seed:
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.benchmark = True #for accelerating the running

        setup_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        dataset_train = list(dataset_train)
        dataset_test = list(dataset_test)

        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            classes_dict = build_classes_dict(args, dataset_train)
            dict_users = sample_dirichlet_train_data(classes_dict,
                                                     args.num_users,  # 100
                                                     alpha=0.5)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar/', train=False, download=True, transform=trans_cifar)
        dataset_train = list(dataset_train)
        dataset_test = list(dataset_test)

        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            classes_dict = build_classes_dict(args, dataset_train)
            dict_users = sample_dirichlet_train_data(classes_dict,
                                                     args.num_users,  # 100
                                                     alpha=0.5)

    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == 'resnet50' and args.dataset == 'cifar':
        net_glob = ResNet50().to(args.device)
    elif args.model == 'lenet5' and args.dataset == 'mnist':
        net_glob = LeNet().to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP().to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = copy.deepcopy(net_glob.state_dict())

    # training
    loss_train = []
    acc_test_list = []
    acc_po_list = []
    current_weights = np.concatenate([i.cpu().data.numpy().flatten() for i in net_glob.parameters()])
    users_grads = np.empty((args.num_users, len(current_weights)), dtype=current_weights.dtype)
    agent_name_keys = [i for i in range(args.num_users)]
    trust_score = [1 for i in range(args.num_users)]
    epochs_submit_update_dict = dict()
    num_samples_dict = dict()
    epoch_images = []
    loss_1 = 1.0
    loss_2 = 1.0

    for iter in range(args.epochs):
        loss_locals = []
        client_w = []
        agent_updates_dict = {}

        for idx in range(args.num_users):
            if idx in args.attacker_list:
                if args.iid:
                    dataset_train_ = dict_users[idx][0]
                else:
                    dataset_train_ = [dataset_train[dict_users[idx][ind]] for ind in range(len(dict_users[idx]))]
                local_t = LocalUpdate(args=args, dataset=dataset_train_, idxs=idx, data_size=len(dataset_train_),
                                      epoch_size=args.epochs)
                if iter < args.attack_start:
                    loss, w, epochs_local_update_list, weight, local_update = local.train(
                        net=copy.deepcopy(net_glob).to(args.device))
                else:
                    loss, w, epochs_local_update_list, weight, local_update = local_t.train_po(
                        net=copy.deepcopy(net_glob).to(args.device))

            else:
                if args.iid:
                    dataset_train_ = dict_users[idx][0]
                else:
                    dataset_train_ = [dataset_train[dict_users[idx][ind]] for ind in range(len(dict_users[idx]))]
                local = LocalUpdate(args=args, dataset=dataset_train_, idxs=idx, data_size=len(dataset_train_),
                                    epoch_size=args.epochs)
                loss, w, epochs_local_update_list, weight, local_update = local.train(net=copy.deepcopy(net_glob).to(args.device))

            client_w.append(w)
            epochs_submit_update_dict[idx] = epochs_local_update_list
            agent_updates_dict[idx] = local_update
            num_samples_dict[idx] = len(dataset_train_)
            users_grads[idx, :] = copy.deepcopy(weight)
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob, trust_score = aggregation(args, copy.deepcopy(net_glob), agent_name_keys, epochs_submit_update_dict,
                                num_samples_dict, users_grads, agent_updates_dict, client_w, dataset_test, epoch_images, iter, trust_score, loss_1, loss_2)

         # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        # testing
        net_glob.eval()
        acc_test, loss_test = test_img(copy.deepcopy(net_glob), dataset_test, args)
        acc_test_po, loss_test_po = test_img_poison(copy.deepcopy(net_glob), dataset_test, args, iter)
        print("Testing accuracy: {:.2f}, Testing Attack Success Rate: {:.2f}".format(acc_test, acc_test_po))
        loss_1 = loss_2
        loss_2 = loss_test
