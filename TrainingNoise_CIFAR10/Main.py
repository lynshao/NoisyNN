import copy, os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from options import args_parser
from Update import LocalUpdate
from Fed import *
import random, time, pickle, math
import pdb, math
import scipy.io

# parse args
args = args_parser()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.arch == 'resnet18':
    from resnet import *
elif args.arch == 'shufflenet':
    from shufflenetv2 import *

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from cifar dataset
    """
    lenRandom = 40000
    num_items = int(lenRandom/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for ii in range(num_users):
        dict_users[ii] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[ii])

    labels = np.array(dataset.targets)
    labels = labels[all_idxs]

    # sort labels
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    numImage = int(len(idxs)/num_users)
    for ii in range(num_users):
        temp = idxs[ii*numImage:(ii+1)*numImage]
        dict_users[ii] = np.concatenate((list(dict_users[ii]), temp), axis=0)

    return dict_users

def testNets(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=False, num_workers=4)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def preprocessing(args):
    # load dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
    
    # split users
    dict_users = cifar_noniid(dataset_train, args.num_users)
    
    # build model
    if args.arch == 'resnet18':
        net_glob = ResNet18().to(args.device)
    elif args.arch == 'shufflenet':
        net_glob = ShuffleNetV2(1).to(args.device)
    
    if args.device == 'cuda':
        net_glob = torch.nn.DataParallel(net_glob)
        torch.backends.cudnn.benchmark = True
    return net_glob, dict_users, dataset_train, dataset_test

def FEEL(args, net_glob, dict_users, dataset_train, dataset_test, AWGNs):
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # keep trace of training loss, achieved accuracies
    loss_train = []
    acc_store = np.array([])

    for iter in np.arange(args.epochs):
        # keep trace of the time consumption of each FEEL iteration
        time1 = time.time()

        # evaluate the global model every 5 itertaions
        if np.mod(iter,5) == 0:
            # testing
            net_glob.eval()
            acc_test, _ = testNets(net_glob, dataset_test, args)
            acc_store = np.append(acc_store, acc_test.numpy())
            print('-' * 40)
            print("Test accuracies =", acc_store)
            print('-' * 40)
            net_glob.train()
        
        # record the starting weights in an iteration
        history_dict = net_glob.state_dict()
        
        loss_locals = []
        w_locals = []
        numactDevices = max(int(args.frac * args.num_users), 1) # num of active devices
        idxs_users = np.random.choice(range(args.num_users), numactDevices, replace=False)

        # learning rate schedule (cosine annealing)
        lrT = args.lr * (1 + math.cos(math.pi * iter  / args.epochs)) / 2
        # local training (PARALLEL -- TBD)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), history_dict=history_dict, lrT= lrT)
            loss_locals.append(copy.deepcopy(loss))
            w_locals.append(copy.deepcopy(w))

        # Uplink model aggregation
        if args.withNoise == 0: # no noise, FL
            current_dict = FedAvg_noNoise(w_locals, args, 0)
        elif args.withNoise == 1: # with noise, FEEL
            current_dict = FEEL_OAC(w_locals, args, AWGNs[iter,:])

        # update global weights
        for k in current_dict.keys():
            w_glob[k] = history_dict[k] + current_dict[k]

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Time Cosumed {:.3f}'.format(iter, loss_avg, time.time()-time1))
        # early stopping if training diverges
        if loss_avg > 20 or math.isnan(loss_avg):
            return np.zeros(21) + 10
        loss_train.append(loss_avg)

    # testing
    acc_test, _ = testNets(net_glob, dataset_test, args)
    acc_store = np.append(acc_store, acc_test.numpy())

    return acc_store

# Grid search under a given SNR, various lambda and beta
if args.arch == 'resnet18':
    lambvec = np.array([0.85, 0.9, 0.95, 1])
    betavec = np.array([0])
    LL = 11167949 # number of parameters
elif args.arch == 'shufflenet':
    lambvec= np.array([0.995, 1])
    betavec = np.array([-0.6, -0.3, 0, 0.3, 0.6])
    LL = 1247674 # number of parameters

# Run experiments 10 times
for eachExp in np.arange(1, 11, 1):
    # load dataset, split users, and build model
    net_glob, dict_users, dataset_train, dataset_test = preprocessing(args)
    # In each run, store the initialized NN weights
    # To ensure that the starting point of grid searches is the same
    initW = copy.deepcopy(net_glob.state_dict())
    
    # In each run, randomly generate the AWGN noise in each epochs first
    # To ensure that the added noise in grid searches is the same
    AWGNs = np.random.normal(0,1,(args.epochs+1, LL))
    
    # Store the grid-search results
    Seqs = np.zeros([len(lambvec), len(betavec), 21])
    # Grid search
    for idx1 in range(len(lambvec)):
        for idx2 in range(len(betavec)):
            startTime = time.time()
            # determine lambda_prime and beta
            args.lamb = lambvec[idx1]
            args.beta = betavec[idx2]
            print("->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->")
            print("->->->->->->->->->->->->->->->-> Running lambda, beta = ",(args.lamb,args.beta))
            net_glob.load_state_dict(initW)
            # Federated edge learning
            outputSeq = FEEL(args, net_glob, dict_users, dataset_train, dataset_test, AWGNs)
            Seqs[idx1, idx2, :] = outputSeq
            print("time = ", time.time() - startTime)
            print(Seqs)
    
    # Store the search results into matlab file to plot 3D figures
    # matplotlib is also possible, but with less functionality
    store_str = './data_' + str(eachExp) + '.mat'
    scipy.io.savemat(store_str,mdict={'Seqs':Seqs})
