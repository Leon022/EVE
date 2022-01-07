#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import h5py
import os
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from models.backdoor_data import MyDataset_cifar, MyDataset_mnist, MyDataset_cifar_DBA, MyDataset_mnist_DBA

def model_dist_norm_var(args, model, target_params_variables, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var= sum_var.to(args.device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

def sum_varance(args, model):
    size = 0
    for name in model.keys():
        size += model[name].view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var = sum_var.to(args.device)
    size = 0
    for name in model.keys():
        sum_var[size:size + model[name].view(-1).shape[0]] = (
                model[name]).view(-1)
        size += model[name].view(-1).shape[0]

    return sum_var

def model_dist_cosine_similarity(args, model, target_params_variables, dim=0):
    net_dict = model.state_dict()
    sum_model = sum_varance(args, net_dict)
    sum_target = sum_varance(args, target_params_variables)
    cs = torch.cosine_similarity(sum_model, sum_target, dim=0)
    loss = abs((1-cs)*10)
    return loss


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, data_size=600, epoch_size=30):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.dataset =dataset
        self.idx = idxs
        self.data_size = data_size
        self.epoch_size = epoch_size

    def train(self, net):
        initial_global_model_params = parameters_to_vector(net.parameters()).detach()
        net.train()

        last_local_model = dict()
        client_grad = []  # only works for aggr_epoch_interval=1
        epochs_local_update_list = []

        for name, data in net.state_dict().items():
            last_local_model[name] = net.state_dict()[name].clone()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                    momentum=0.9, weight_decay=5e-4)

        ldr_train = DataLoader(self.dataset, batch_size=self.args.local_bs, shuffle=True)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            target_params_variables = dict()
            for name, param in net.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs, _ = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                if self.args.aggregation_methods == 'foolsgold':
                    for i, (name, params) in enumerate(net.named_parameters()):
                        if params.requires_grad:
                            if iter == 0 and batch_idx == 0:
                                client_grad.append(params.grad.clone())
                            else:
                                client_grad[i] += params.grad.clone()

                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            local_model_update_dict = dict()
            for key, data in net.state_dict().items():
                local_model_update_dict[key] = torch.zeros_like(data)
                local_model_update_dict[key] = (data - last_local_model[key])
                last_local_model[key] = copy.deepcopy(data)

            if self.args.aggregation_methods == 'foolsgold':
                epochs_local_update_list.append(client_grad)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        weight = np.float64(np.concatenate([param.data.cpu().numpy().flatten() for param in net.parameters()]))
        update = parameters_to_vector(net.parameters()).double() - initial_global_model_params

        return sum(epoch_loss)/len(epoch_loss), net.state_dict(), epochs_local_update_list, weight, update


    def train_po(self, net):
        initial_global_model_params = parameters_to_vector(net.parameters()).detach()
        net.train()

        last_local_model = dict()
        client_grad = []  # only works for aggr_epoch_interval=1
        epochs_local_update_list = []

        for name, data in net.state_dict().items():
            last_local_model[name] = net.state_dict()[name].clone()

        distance_params_variables = dict()
        for name, param in net.named_parameters():
            distance_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr_ba,
                                    momentum=0.9, weight_decay=5e-4)
        if self.args.dataset == 'cifar':
            if self.args.attack_methods == 'CBA':
                dataset_poison = MyDataset_cifar(self.dataset, self.args.target_label, portion=self.args.back_prop, mode="train")
            elif self.args.attack_methods == 'DBA':
                dataset_poison = MyDataset_cifar_DBA(self.args, self.dataset, self.args.target_label, portion=self.args.back_prop, mode="train", idx=self.idx)
        elif self.args.dataset == 'mnist':
            if self.args.attack_methods == 'CBA':
                dataset_poison = MyDataset_mnist(self.dataset, self.args.target_label, portion=self.args.back_prop, mode="train")
            elif self.args.attack_methods == 'DBA':
                dataset_poison = MyDataset_mnist_DBA(self.args, self.dataset, self.args.target_label, portion=self.args.back_prop, mode="train", idx=self.idx)

        ldr_train = DataLoader(dataset_poison, batch_size=self.args.local_bs_ba, shuffle=True)
        epoch_loss = []
        for iter in range(self.args.local_ep_ba):
            target_params_variables = dict()
            for name, param in net.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs, _ = net(images)
                # labels = torch.argmax(labels, dim=1)
                class_loss = self.loss_func(log_probs, labels.long())
                distance_loss = model_dist_norm_var(self.args, net, distance_params_variables)
                loss = class_loss*0.8 + distance_loss*0.2
                loss.backward()

                if self.args.aggregation_methods == 'foolsgold':
                    for i, (name, params) in enumerate(net.named_parameters()):
                        if params.requires_grad:
                            if iter == 0 and batch_idx == 0:
                                client_grad.append(params.grad.clone())
                            else:
                                client_grad[i] += params.grad.clone()

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


            local_model_update_dict = dict()
            for key, data in net.state_dict().items():
                local_model_update_dict[key] = torch.zeros_like(data)
                local_model_update_dict[key] = (data - last_local_model[key])
                last_local_model[key] = copy.deepcopy(data)

            if self.args.aggregation_methods == 'foolsgold':
                epochs_local_update_list.append(client_grad)
            else:
                epochs_local_update_list.append(local_model_update_dict)

        weight = np.float64(np.concatenate([param.data.cpu().numpy().flatten() for param in net.parameters()]))
        update = parameters_to_vector(net.parameters()).double() - initial_global_model_params

        return sum(epoch_loss) / len(epoch_loss), net.state_dict(), epochs_local_update_list, weight, update