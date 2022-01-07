#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import copy

import numpy as np
# import cv2
from utils.options import args_parser
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

import matplotlib.pyplot as plt
#忽略警告
import warnings
warnings.filterwarnings('ignore')


class MyDataset_mnist(Dataset):

    def __init__(self, dataset, target, portion=0.5, mode="train", device=args.device):
        self.dataset = self.addTrigger(dataset, target, portion, mode)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (1, 28, 28))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 2.82
            distance = 3
            if i in perm:
                for c in range(channels):
                    img[c, width - distance, height - distance] = pixel_value
                    img[c, width - distance - 1, height - distance - 1] = pixel_value
                    img[c, width - distance, height - distance - 2] = pixel_value
                    img[c, width - distance - 2, height - distance] = pixel_value

                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_

class MyDataset_cifar(Dataset):
    def __init__(self, dataset, target, portion=0.5, mode="train", device=args.device):
        self.dataset = self.addTrigger(dataset, target, portion, mode)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()

        # img = img[..., np.newaxis]
        #img = np.expand_dims(img, axis=0)
        # img = torch.Tensor(img).permute(2,0,1)
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)


    def addTrigger(self, dataset, target, portion, mode):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (3, 32, 32))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 1
            distance = 3
            if i in perm:
                for c in range(channels):
                    img[c, width - distance, height - distance] = pixel_value
                    img[c, width - distance - 1, height - distance - 1] = pixel_value
                    img[c, width - distance, height - distance - 2] = pixel_value
                    img[c, width - distance - 2, height - distance] = pixel_value

                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_


class MyDataset_mnist_DBA(Dataset):

    def __init__(self, args, dataset, target, portion=0.5, epoch=1, mode="train", idx=2, device=args.device):
        self.dataset = self.addTrigger(args, dataset, target, portion, mode, idx, epoch)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, args, dataset, target, portion, mode, idx, epoch):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (1, 28, 28))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 2.82
            distance = 3
            if i in perm:
                for c in range(channels):
                    if idx==args.attacker_list[0]:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                    elif idx==args.attacker_list[1]:
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                    elif idx==args.attacker_list[2]:
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                    elif idx==args.attacker_list[3]:
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value
                    else:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value


                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_


class MyDataset_cifar_DBA(Dataset):
    def __init__(self, args, dataset, target, portion=0.5, mode="train", idx=2, device=args.device):
        self.dataset = self.addTrigger(args, dataset, target, portion, mode, idx)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]

        # img1 = img.transpose(1, 2, 0)
        # plt.figure(1)
        # plt.imshow(img1)
        # plt.show()

        # img = img[..., np.newaxis]
        #img = np.expand_dims(img, axis=0)
        # img = torch.Tensor(img).permute(2,0,1)
        img = torch.Tensor(img)

        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        label = torch.argmax(label, -1)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)


    def addTrigger(self, args, dataset, target, portion, mode, idx):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            # print(np.max(img))
            img = np.resize(img, (3, 32, 32))
            channels = img.shape[0]
            width = img.shape[1]
            height = img.shape[2]
            pixel_value = 1
            distance = 3
            if i in perm:
                for c in range(channels):
                    if idx == args.attacker_list[0]:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                    elif idx == args.attacker_list[1]:
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                    elif idx == args.attacker_list[2]:
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                    elif idx == args.attacker_list[3]:
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value
                    else:
                        img[c, 0, 0] = pixel_value
                        img[c, 0, 1] = pixel_value
                        img[c, 0, 2] = pixel_value
                        img[c, 0, 3] = pixel_value
                        img[c, 0, 6] = pixel_value
                        img[c, 0, 7] = pixel_value
                        img[c, 0, 8] = pixel_value
                        img[c, 0, 9] = pixel_value
                        img[c, 3, 0] = pixel_value
                        img[c, 3, 1] = pixel_value
                        img[c, 3, 2] = pixel_value
                        img[c, 3, 3] = pixel_value
                        img[c, 3, 6] = pixel_value
                        img[c, 3, 7] = pixel_value
                        img[c, 3, 8] = pixel_value
                        img[c, 3, 9] = pixel_value

                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_