import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os, argparse, pdb, random, time, scipy.io
from f_add_noise_and_decode import Add_noise, denoiser

# Input
parser = argparse.ArgumentParser(description='PyTorch NoisyNN CIFAR-10')
parser.add_argument('--denoiser', default=2, type=int, help='ML or MMSEpb denoiser')
parser.add_argument('--EsN0dB', default=-20., type=float, help='EsN0 in dB')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('--kappa', type=int, default=100)
args = parser.parse_args()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.arch == 'resnet18' or args.arch == 'resnet34':
    from resnet import *
elif args.arch == 'shufflenet':
    from shufflenetv2 import ShuffleNetV2


# Prepare CIFAR-10 data
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(trainset, batch_size=args.kappa, shuffle=False, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

def testModel(net):
    net.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    with torch.no_grad():
        # compute the validation accuracy
        for batch_idx, (inputs, targets) in enumerate(validloader):
            if batch_idx >= 1:
                break
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total2 += targets.size(0)
            correct2 += predicted.eq(targets).sum().item()

        # compute the test accuracy
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total1 += targets.size(0)
            correct1 += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc1 = 100.*correct1/total1
    acc2 = 100.*correct2/total2
    return acc1, acc2

# Generate NN
if args.arch == 'resnet18':
    net = ResNet18()
    checkpoint = torch.load('./ResNet18_pretrained.pth')
elif args.arch == 'resnet34':
    net = ResNet34()
    checkpoint = torch.load('./ResNet34_pretrained.pth')
elif args.arch == 'shufflenet':
    net = ShuffleNetV2(1)
    checkpoint = torch.load('./ShuffleNet_pretrained.pth')

net = net.to(args.device)
if args.device == 'cuda':
    net = torch.nn.DataParallel(net)
    torch.backends.cudnn.benchmark = True

weights = checkpoint['net']

# Grid search under a given SNR, various lambda_prime and beta
if args.arch == 'resnet18':
    lambvec = np.arange(0.95, 1.07, 0.01)
    betavec = np.arange(-4, 0.51, 0.2)
elif args.arch == 'resnet34':
    lambvec = np.arange(0.95, 1.07, 0.01)
    betavec = np.arange(-4, 0.51, 0.2)
elif args.arch == 'shufflenet':
    lambvec = np.arange(0.9, 1.16, 0.01)
    betavec = np.arange(-4, 4.1, 0.5)

# Run experiments 10 times
for eachExp in np.arange(1, 11, 1):
    # In each run, generate the NoisyNN first
    noisyWeights, varW, varN, StoreRecover = Add_noise(weights, args)
    
    # Grid search
    for idx1 in range(len(lambvec)):
        oneSeq1 = np.array([])
        oneSeq2 = np.array([])
        for idx2 in range(len(betavec)):
            startTime = time.time()
            # determine lambda_prime and beta
            args.lamb = lambvec[idx1]
            args.beta = betavec[idx2]
            print("-->-->-->-- Running lambda, beta = ",(args.lamb,round(args.beta,4)))
            # MMSEpb denoising
            denoisedWeights = denoiser(weights, args, noisyWeights, varW, varN, StoreRecover)
            net.load_state_dict(denoisedWeights)
            accuracy1, accuracy2 = testModel(net)
            oneSeq1 = np.append(oneSeq1, accuracy1)
            oneSeq2 = np.append(oneSeq2, accuracy2)
            print("eachExp, Accuracy, Time = ",(eachExp, accuracy1, round(time.time() - startTime,4)))

        if idx1 == 0:
            Seqs1 = np.array([oneSeq1])
            Seqs2 = np.array([oneSeq2])
        else:
            Seqs1 = np.r_[Seqs1,np.array([oneSeq1])]
            Seqs2 = np.r_[Seqs2,np.array([oneSeq2])]

    # Store the search results into matlab file to plot 3D figures
    # matplotlib is also possible, but with less functionality
    store_str1 = './testdata_' + str(eachExp) + '.mat'
    scipy.io.savemat(store_str1,mdict={'Seqs1':Seqs1})
    store_str2 = './validdata_' + str(eachExp) + '.mat'
    scipy.io.savemat(store_str2,mdict={'Seqs2':Seqs2})
