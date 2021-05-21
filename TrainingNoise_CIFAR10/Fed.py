#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, pdb, math, time
import numpy as np
import torch
from functools import partial


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
def FEEL_OAC(w, args, AWGN):
    """
    Noised is added in the following way: Tensor -> numpy (add noise) -> Tensor.
    Noise directly added to Tensors (TBD)
    """
    MM = len(w) # number of devices

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

    if args.Estimator == 1: # ML
        output = RecevidSignal
    elif args.Estimator == 2:  # MMSEpb estimation
        varW = np.var(SignalPart)
        eta = varW/varN
        cc = eta/(eta + 1 - args.lamb)
        dd = varW * args.beta /(eta + 1 - args.lamb)
        output = cc * RecevidSignal + dd
    else:
        print("Error: unknwon estimator!")
    
    # Averaging the batch-normalization layers
    w_avg = FedAvg_noNoise(w, args, 1)
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

    
    return w_avg