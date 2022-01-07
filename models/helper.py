#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import torch

from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.functional import log_softmax
import torch.nn.functional as F
from models.nets import LeNet, MLP, ResNet18, ResNet50
import time

logger = logging.getLogger("logger")
import os
import json
import numpy as np
import copy
from utils.options import args_parser

args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


def init_weight_accumulator(target_model):
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(data)

    return weight_accumulator


def accumulate_weight(args, weight_accumulator, epochs_submit_update_dict, state_keys, num_samples_dict):
    """
     return Args:
         updates: dict of (num_samples, update), where num_samples is the
             number of training samples corresponding to the update, and update
             is a list of variable weights
     """
    if args.aggregation_methods == 'foolsgold':
        updates = dict()
        for i in range(0, len(state_keys)):
            local_model_gradients = epochs_submit_update_dict[state_keys[i]][0]  # agg 1 interval
            num_samples = num_samples_dict[state_keys[i]]
            updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
        return None, updates

    else:
        updates = dict()
        for i in range(0, len(state_keys)):
            local_model_update_list = epochs_submit_update_dict[state_keys[i]]
            update = dict()
            num_samples = num_samples_dict[state_keys[i]]

            for name, data in local_model_update_list[0].items():
                update[name] = torch.zeros_like(data)

            for j in range(0, len(local_model_update_list)):
                local_model_update_dict = local_model_update_list[j]
                for name, data in local_model_update_dict.items():
                    weight_accumulator[name].add_(local_model_update_dict[name])
                    update[name].add_(local_model_update_dict[name])
                    detached_data = data.cpu().detach().numpy()
                    # print(detached_data.shape)
                    detached_data = detached_data.tolist()
                    # print(detached_data)
                    local_model_update_dict[name] = detached_data  # from gpu to cpu

            updates[state_keys[i]] = (num_samples, update)

    return weight_accumulator, updates

def FedAvg(net, w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def average_shrink_models(weight_accumulator, target_model, epoch_interval):
    """
    Perform FedAvg algorithm and perform some clustering on top of it.

    """
    for name, data in target_model.state_dict().items():
        # if self.params.get('tied', False) and name == 'decoder.weight':
        #     continue

        eta=1
        update_per_layer = weight_accumulator[name] * (eta / (args.num_users))
        # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

        # update_per_layer = update_per_layer * 1.0 / epoch_interval
        # if self.params['diff_privacy']:
        #     update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
        data = data.float()
        update_per_layer = update_per_layer.float()
        data.add_(update_per_layer)

    return True, copy.deepcopy(target_model.state_dict())

def geometric_median_update(target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm= None):
    """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
           """
    points = []
    alphas = []
    names = []
    for name, data in updates.items():
        points.append(data[1]) # update
        alphas.append(data[0]) # num_samples
        names.append(name)

    alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
    alphas = torch.from_numpy(alphas).float()

    # alphas.float().to(config.device)
    median = weighted_average_oracle(points, alphas)
    num_oracle_calls = 1

    # logging
    obj_val = geometric_median_objective(median, points, alphas)
    logs = []
    log_entry = [0, obj_val, 0, 0]
    logs.append(log_entry)
    # start
    wv=None
    for i in range(maxiter):
        prev_median, prev_obj_val = median, obj_val
        weights = torch.tensor([alpha / max(eps, l2dist(median, p)) for alpha, p in zip(alphas, points)],
                             dtype=alphas.dtype)
        weights = weights / weights.sum()
        median = weighted_average_oracle(points, weights)
        num_oracle_calls += 1
        obj_val = geometric_median_objective(median, points, alphas)
        log_entry = [i + 1, obj_val,
                     (prev_obj_val - obj_val) / obj_val,
                     l2dist(median, prev_median)]
        logs.append(log_entry)
        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break
        wv=copy.deepcopy(weights)
    alphas = [l2dist(median, p) for p in points]

    update_norm = 0
    for name, data in median.items():
        update_norm += torch.sum(torch.pow(data, 2))
    update_norm= math.sqrt(update_norm)

    eta = 0.1
    if max_update_norm is None or update_norm < max_update_norm:
        for name, data in target_model.state_dict().items():
            update_per_layer = median[name] * (eta)
            # if self.params['diff_privacy']:
            #     update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
            data.add_(update_per_layer)
        is_updated = True
    else:
        is_updated = False

    return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(), alphas, copy.deepcopy(target_model.state_dict())


def weighted_average_oracle(points, weights):
    """Computes weighted average of atoms with specified weights

    Args:
        points: list, whose weighted average we wish to calculate
            Each element is a list_of_np.ndarray
        weights: list of weights of the same length as atoms
    """
    tot_weights = torch.sum(weights)

    weighted_updates= dict()

    for name, data in points[0].items():
        weighted_updates[name]=  torch.zeros_like(data)
    for w, p in zip(weights, points): # 对每一个agent
        for name, data in weighted_updates.items():
            temp = (w / tot_weights).float().to(args.device)
            temp= temp* (p[name].float())
            # temp = w / tot_weights * p[name]
            if temp.dtype!=data.dtype:
                temp = temp.type_as(data)
            data.add_(temp)

    return weighted_updates

def l2dist(p1, p2):
    """L2 distance between p1, p2, each of which is a list of nd-arrays"""
    squared_sum = 0
    for name, data in p1.items():
        squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
    return math.sqrt(squared_sum)

def geometric_median_objective(median, points, alphas):
    """Compute geometric median objective."""
    temp_sum= 0
    for alpha, p in zip(alphas, points):
        temp_sum += alpha * l2dist(median, p)
    return temp_sum

    # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

def model_dist_norm_var(model, target_params_variables, norm=2):
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

def foolsgold_update(target_model, updates):
    client_grads = []
    alphas = []
    names = []
    eta = 0.1
    for name, data in updates.items():
        client_grads.append(data[1])  # gradient
        alphas.append(data[0])  # num_samples
        names.append(name)

    target_model.train()
    # train and update
    optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1,
                                momentum=0.9,
                                weight_decay=0.0005)

    optimizer.zero_grad()
    fg = FoolsGold(use_memory=True)
    agg_grads, wv,alpha = fg.aggregate_gradients(client_grads,names)
    for i, (name, params) in enumerate(target_model.named_parameters()):
        agg_grads[i]=agg_grads[i] * eta
        if params.requires_grad:
            params.grad = agg_grads[i].to(args.device)
    optimizer.step()
    wv=wv.tolist()
    return True, names, wv, alpha, copy.deepcopy(target_model.state_dict())


class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def foolsgold(self,grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha

import functools
from collections import defaultdict

def row_into_parameters(row, grad):
    offset = 0
    for name in grad.keys():
        new_size = functools.reduce(lambda x,y:x*y, grad[name].shape)
        current_data = row[offset:offset + new_size]

        grad[name][:] = torch.from_numpy(current_data.reshape(grad[name].shape))
        offset += new_size
    return grad


def trimmed_mean(users_grads, users_count, corrupted_count):
    number_to_consider = int(users_count - corrupted_count)
    current_grads = np.empty((users_grads.shape[1],), users_grads.dtype)

    for i, param_across_users in enumerate(users_grads.T):
        med = np.median(param_across_users)
        good_vals = sorted(param_across_users - med, key=lambda x: abs(x))[:number_to_consider]
        current_grads[i] = np.mean(good_vals) + med
    return current_grads

def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j], ord=1)
    return distances

def krum(users_grads, users_count, corrupted_count, distances=None,return_index=False, debug=False):
    if not return_index:
        assert users_count >= 2*corrupted_count + 1,('users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        print(users_grads[minimal_error_index])
        return users_grads[minimal_error_index]

def bulyan(users_grads, users_count, corrupted_count):
    assert users_count >= 4*corrupted_count + 3
    set_size = users_count - corrupted_count
    selection_set = []

    distances = _krum_create_distances(users_grads)
    while len(selection_set) < set_size:
        currently_selected = krum(users_grads, users_count - len(selection_set), corrupted_count, distances, True)
        selection_set.append(users_grads[currently_selected])

        # remove the selected from next iterations:
        distances.pop(currently_selected)
        for remaining_user in distances.keys():
            distances[remaining_user].pop(currently_selected)

    return trimmed_mean(np.array(selection_set), len(selection_set), 2*corrupted_count)


class Aggregation():
    def __init__(self, agent_data_sizes, n_params, args):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.server_lr = 1
        self.n_params = n_params
        self.cum_net_mov = 0


    def aggregate_updates(self, global_model, agent_updates_dict):
        # adjust LR if robust LR is selected
        lr_vector = torch.Tensor([self.server_lr] * self.n_params).to(self.args.device)
        if self.args.robustLR_threshold > 0:
            lr_vector = self.compute_robustLR(agent_updates_dict)

        aggregated_updates = self.agg_avg(agent_updates_dict)
        cur_global_params = parameters_to_vector(global_model.parameters())
        new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        vector_to_parameters(new_global_params, global_model.parameters())

        return global_model.state_dict()

    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.args.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.args.device)

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data
        return sm_updates / total_data

from tqdm import tqdm
from torch import nn
from models.clustering import kmeans

def pgd_attack(model, images, labels, eps=0.5, alpha=2 / 255, iters=40):
    images = images.to(args.device)
    labels = labels.to(args.device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data
    cost_t = 0

    for i in range(iters):
        images.requires_grad = True
        outputs, _ = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(args.device)
        cost_t += cost
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images, cost_t

def fl_eve(args, glob_model, client_w, trust_score, epoch, detection_data, epoch_images, loss_1, loss_2):
    print("Start testing!")
    last_ac_list = []
    delta_loss = (loss_2 - loss_1) / loss_1
    dataset_taregt = []
    for i in tqdm(range(len(detection_data))):
        data = detection_data[i]
        if data[1] == args.defense_label:
            dataset_taregt.append((data[0], data[1]))

    def testing(net, idx):
        if args.model == 'resnet18' and args.dataset == 'cifar':
            model = ResNet18().to(args.device)
        elif args.model == 'resnet50' and args.dataset == 'cifar':
            model = ResNet50().to(args.device)
        elif args.model == 'lenet5' and args.dataset == 'mnist':
            model = LeNet().to(args.device)
        elif args.model == 'mlp' and args.dataset == 'mnist':
            model = MLP().to(args.device)
        else:
            exit('Error: unrecognized model')
        model.load_state_dict(net)
        model.to(args.device)

        model.eval()
        dataloader = torch.utils.data.DataLoader(dataset_taregt[:args.detection_size], batch_size=args.detection_size, shuffle=True, num_workers=2)
        correct = 0
        total = 0
        loss_ = nn.CrossEntropyLoss()
        activation_ = []

        for images, labels in dataloader:
            if epoch<=1:
                images, cost = pgd_attack(model, images, labels, args.eps)
                epoch_images.append(images)
            elif delta_loss>=args.gamma and epoch>1:
                images, cost = pgd_attack(model, images, labels, args.eps)
                epoch_images[idx] = images
            else:
                images = epoch_images[idx]

            labels = labels.to(args.device)
            outputs, ac = model(images)
            activation_.append(ac)
            _, pre = torch.max(outputs.data, 1)
            total += 1
            correct += (pre == labels).sum()

        ac_total = activation_[0]
        d = ac_total[0]
        for k in range(1, ac_total.shape[0]):
            d += ac_total[k]
        d = d.cpu().detach().numpy()

        return d

    gl_ac = testing(glob_model, idx=args.num_users)

    for i in range(len(client_w)):
        ac_total = testing(client_w[i], i)
        last_ac_list.append(ac_total)

    last_ac_list.append(gl_ac)
    centroids, cluster = kmeans(last_ac_list, args.k)

    res = []
    for a in range(args.k):
        index = []
        for b in range(len(cluster[a])):
            for c in range(args.num_users+1):
                if (cluster[a][b] == last_ac_list[c]).all():
                    index.append(c)
                    break
        res.append(index)
    print(res)


    for g in range(len(res)):
        if args.num_users in res[g]:
            print(res[g])
            if len(res[g]) == 1:
                w_avg = dict()
                total_score = 0
                for gg in range(len(trust_score)):
                    total_score += trust_score[gg]
                for key in glob_model.keys():
                    w_avg[key] = torch.zeros_like(glob_model[key])
                    for i in range(len(client_w)):
                        w_avg[key] += ((client_w[i][key] - glob_model[key]).float() * (trust_score[i] / total_score))

                for k in range(len(trust_score)):
                    trust_score[k] += 1

            else:
                res[g].remove(args.num_users)
                w_avg = dict()
                total_score = 0
                for gg in range(len(res[g])):
                    total_score += trust_score[res[g][gg]]
                for key in glob_model.keys():
                    w_avg[key] = torch.zeros_like(glob_model[key])
                    for i in range(len(res[g])):
                        w_avg[key] += ((client_w[res[g][i]][key] - glob_model[key]).float() * (
                                trust_score[res[g][i]] / total_score))

                for k in range(len(res[g])):
                    trust_score[res[g][k]] += 1

    for key in glob_model.keys():
        glob_model[key] += w_avg[key]

    return glob_model, trust_score
