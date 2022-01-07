#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import defaultdict
import random

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = [dataset[num_items * i: num_items * (i + 1)]]

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = [dataset[num_items * i: num_items * (i + 1)]]
    return dict_users

def build_classes_dict(args, train_dataset):
    cifar_classes = {}
    data_num_avg = int(len(train_dataset)/args.num_users)
    for ind, x in enumerate(train_dataset):
        if args.dataset == 'mnist':
            if ind<=(args.num_users*data_num_avg):   #
                _, label = x
                if label in cifar_classes:
                    cifar_classes[label].append(ind)
                else:
                    cifar_classes[label] = [ind]
        elif args.dataset == 'cifar':
            if ind<=(args.num_users*data_num_avg):
                _, label = x
                if label in cifar_classes:
                    cifar_classes[label].append(ind)
                else:
                    cifar_classes[label] = [ind]
        else:
            exit('Error: unrecognized dataset')
    return cifar_classes

def sample_dirichlet_train_data(classes_dict, no_participants, alpha=0.9):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """
    cifar_classes = classes_dict
    class_size = len(cifar_classes[0]) #for cifar: 5000
    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())  # for cifar: 10

    image_nums = []
    for n in range(no_classes):
        image_num = []
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            image_num.append(len(sampled_list))
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        image_nums.append(image_num)
    # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
    return per_participant_list

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_iid(dataset_train, num)