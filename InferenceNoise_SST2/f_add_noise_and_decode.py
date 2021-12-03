import copy, pdb
import numpy as np
import torch
from functools import partial

def flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)


def Add_noise(weights, args):
    """
    Noised is added in the following way: Tensor -> numpy (add noise) -> Tensor.
    Noise directly added to Tensors (TBD)
    """
    # Used for Tensor recovering
    StoreRecover = np.array([])

    # weights in numpy 
    SignalPart = np.array([])
    for k in weights.keys():
        # bypass layer-normalization layers
        if ('position_ids' in  k) or ('LayerNorm' in k):
            continue
        temp = weights[k].cpu().numpy()
        temp, unflatten = flatten(temp)
        StoreRecover = np.append(StoreRecover,unflatten)
        SignalPart = np.append(SignalPart, temp)

    LL = len(SignalPart)

    # compute the received signal power and noise power (for a given EsN0dB)
    SigPower = np.sum(np.power(np.abs(SignalPart),2))/LL
    EsN0 = np.power(10, args.EsN0dB/10.0)
    varN = SigPower/EsN0
    
    # noisy NN weights
    noisyWeights = SignalPart + np.random.normal(0,1,LL) * np.sqrt(varN)
    varW = np.var(SignalPart)

    return noisyWeights, varW, varN, StoreRecover


def denoiser(weights, args, noisyWeights, varW, varN, StoreRecover):
    # denoise the NoisyNN
    if args.denoiser == 1: # ML estimator 
        output = noisyWeights
    elif args.denoiser == 2: # MMSEpb denoiser
        eta = varW/varN
        cc = eta/(eta + 1 - args.lamb)
        dd = varW * args.beta /(eta + 1 - args.lamb)
        output = cc * noisyWeights + dd
    else:
        print("Error: unknwon denoiser!")
    
    # Transform Numpy -> Tensor
    startIndex = 0
    idx = 0
    w_avg = copy.deepcopy(weights)
    for k in w_avg.keys():
        flag = 1
        # bypass layer-normalization layers
        if ('position_ids' in  k) or ('LayerNorm' in k):
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

