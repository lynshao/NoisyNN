#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, pdb, math, time
import numpy as np
import torch
from functools import partial
from torch.utils.data import DataLoader
import torch.nn.functional as F


def testNets(net_g, datatest, args, flag_valid = 0):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    if flag_valid == 0:
        data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False, num_workers=4)
    else:
        data_loader = DataLoader(datatest, batch_size=args.num, shuffle=False)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if flag_valid == 0 or (flag_valid == 1 and idx == 0):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        else:
            continue

    test_loss /= len(data_loader.dataset)
    if flag_valid == 0:
        accuracy = 100.00 * correct / len(data_loader.dataset)
    else:
        accuracy = 100.00 * correct / args.num
    return accuracy, test_loss

def flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)

# Federated Averaging
def FedAvg_noNoise(w, args, flag = 0):
    """
    if flag = 0, this function averages all layers
    if flag = 1, this function averages the batch-normalization layers
    """
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        # if flag = 1, average the batch-normalization layers only ('shortcut:1' is also a batch-normalization layer)
        if (flag == 1) and ('bn' not in k) and ('shortcut:1' not in k):
            continue
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# Federated edge learning with over-the-air computation
def FEEL_OAC(iter, w, args, AWGN, history_dict, net_glob, dataset_valid):
    """
    Noised is added in the following way: Tensor -> numpy (add noise) -> Tensor.
    Noise directly added to Tensors (TBD)
    """
    MM = len(w) # number of devices    
    # Averaging the batch-normalization layers
    w_avg = FedAvg_noNoise(w, args, 1)

    # Used for Tensor recovering
    StoreRecover = np.array([])
    for idx in np.arange(MM):
        wEach = w[idx]
        eachWeightNumpy = np.array([])
        for k in wEach.keys():
            # bypass batch normalization layers ('shortcut.1' in ResNets is also a batch normalization layer)
            if ('bn' in k) or ('shortcut:1' in k):
                continue
            temp = wEach[k].cpu().numpy()
            temp, unflatten = flatten(temp)
            if idx == 0:
                StoreRecover = np.append(StoreRecover,unflatten)
            eachWeightNumpy = np.append(eachWeightNumpy, temp)

        if idx == 0:
            TransmittedSymbols = np.array([eachWeightNumpy])
        else:
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([eachWeightNumpy])]
    
    # number of symbols to be transmitted
    LL = len(TransmittedSymbols[0])

    # received signal
    SignalPart = np.sum(TransmittedSymbols,0)
    # received signal power
    SigPower = np.sum(np.power(np.abs(SignalPart),2))/LL
    EsN0 = np.power(10, args.EsN0dB/10.0)
    # noise power
    varN = SigPower/EsN0

    # add noise
    RecevidSignal = SignalPart + AWGN * np.sqrt(varN)

    if args.denoiser == 1: # ML
        output = RecevidSignal
    elif args.denoiser == 2:  # MMSEpb
        varW = np.var(SignalPart)
        eta = varW/varN
        # search on validation dataset
        acc_store = np.zeros([len(args.lambvec), len(args.betavec)])
        for idx1 in range(len(args.lambvec)):
            for idx2 in range(len(args.betavec)):
                theta = eta/(eta + 1 - args.lambvec[idx1])
                rho = varW * args.betavec[idx2] /(eta + 1 - args.lambvec[idx1])
                output = theta * RecevidSignal + rho
                newW = recover_DNN(output, w_avg, MM, StoreRecover)
                w_glob = copy.deepcopy(history_dict)
                # update global weights
                for k in newW.keys():
                    w_glob[k] = history_dict[k] + newW[k]

                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)
                # test the model
                acc_test, _ = testNets(net_glob, dataset_valid, args, 0)
                acc_store[idx1, idx2] = acc_test.numpy()

        loc1, loc2 = np.where(acc_store==np.max(acc_store))
        args.loc1 = loc1[0]
        args.loc2 = loc2[0]
        print('chosen locs:', (args.loc1, args.loc2))
        theta = eta/(eta + 1 - args.lambvec[args.loc1])
        rho = varW * args.betavec[args.loc2] /(eta + 1 - args.lambvec[args.loc1])
        output = theta * RecevidSignal + rho
        print(acc_store)
    else:
        print("Error: unknwon denoiser!")

    newW = recover_DNN(output, w_avg, MM, StoreRecover)
    return newW

def recover_DNN(output, w_avg, MM, StoreRecover):
    # average other layers
    output = output/MM

    # Transform Numpy -> Tensor
    startIndex = 0
    idx = 0
    for k in w_avg.keys():
        flag = 1
        # bypass layer-normalization layers
        if ('bn' in k) or ('shortcut:1' in k):
            flag = 0

        if flag == 1:
            lenLayer = w_avg[k].numel()
            # get data
            ParamsLayer = output[startIndex:(startIndex+lenLayer)]
            # reshape
            ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
            # convert to torch in cuda()
            w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()

            startIndex += lenLayer
            idx += 1
    return copy.deepcopy(w_avg)