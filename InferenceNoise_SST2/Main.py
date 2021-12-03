import os
from sys import platform
import argparse, pdb, random, time, scipy.io

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from models import BertModel
from data_sst2 import DataPrecessForSentence
from f_add_noise_and_decode import Add_noise, denoiser


parser = argparse.ArgumentParser(description='PyTorch NoisyNN')
parser.add_argument('--denoiser', default=2, type=int, help='ML or MMSEpb denoiser')
parser.add_argument('--EsN0dB', default=9.0, type=float, help='EsN0 in dB')
parser.add_argument('--kappa', default=1, type=int)
args = parser.parse_args()

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

def testModel(model, dev_loader, test_loader):
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    accuracy1 = 0.0
    # Deactivate autograd for evaluation.
    accuracy1 = 0
    accuracy2 = 0
    with torch.no_grad():
        # return validation accuracy
        cnt = 0
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dev_loader:
            if cnt >= 1:
                break
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            _, predicted = probabilities.max(1)
            accuracy2 += predicted.eq(labels).sum().item()
            cnt += 1

        # return test accuracy
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in test_loader:
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            _, predicted = probabilities.max(1)
            accuracy1 += predicted.eq(labels).sum().item()
        

    accuracy1 /= (len(test_loader.dataset))
    accuracy2 /= args.kappa

    return accuracy1, accuracy2

def main():    
    # ================================================================== Gen NN
    print("\t* Building model...")
    bertmodel = BertModel(requires_grad = False)
    tokenizer = bertmodel.tokenizer
    net = bertmodel.to(args.device)

    # ================================================================== load test data
    print("\t* Loading test data...")
    test_df = pd.read_csv("./test.tsv",sep='\t',header=None, names=['similarity','s1'])
    test_data = DataPrecessForSentence(tokenizer,test_df, max_seq_len = 50) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=32)
    dev_df = pd.read_csv(os.path.join('./',"dev.tsv"),sep='\t',header=None, names=['similarity','s1'])
    dev_data = DataPrecessForSentence(tokenizer,dev_df, max_seq_len = 50)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.kappa)


    # ================================================================== Load already-trained NN weights
    target_dir = "../"
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(os.path.join(target_dir, "bert_pretrained.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join(target_dir, "bert_pretrained.pth.tar"), map_location=device)
        
    weights = checkpoint["model"]

    # =============================== For a fixed EsN0dB, grid search using various combinations of lambda' and beta
    lambvec = np.arange(0.1, 1.11, 0.05)
    lambvec = np.concatenate([[0.02, 0.04, 0.06, 0.08], lambvec])
    betavec = np.arange(-10, 51, 2)
    betavec = np.concatenate([[-30, -20, -15], betavec])
    print("\t* Grid search...")
    print("lambda_prime = ", lambvec)
    print("beta = ", betavec)

    # repeat 10 experiments
    for expInx in np.arange(1, 11, 1):
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
                print("-->-->-->-- Running lambda_prime, beta = ",(args.lamb, round(args.beta,4)))
                # MMSEpb denoiser
                denoisedWeights = denoiser(weights, args, noisyWeights, varW, varN, StoreRecover)
                # load denoised weights and compute inference accuracy
                net.load_state_dict(denoisedWeights)
                accuracy1, accuracy2 = testModel(net, dev_loader, test_loader)
                oneSeq1 = np.append(oneSeq1, accuracy1)
                oneSeq2 = np.append(oneSeq2, accuracy2)
                print("expInx, Accuracy, Time = ",(expInx, accuracy1, round(time.time() - startTime,4)))

            if idx1 == 0:
                Seqs1 = np.array([oneSeq1])
                Seqs2 = np.array([oneSeq2])
            else:
                Seqs1 = np.r_[Seqs1,np.array([oneSeq1])]
                Seqs2 = np.r_[Seqs2,np.array([oneSeq2])]
        
        # Store the search results into matlab file to plot 3D figures
        # matplotlib is also possible, but with less functionality
        store_str1 = './testdata_' + str(expInx) + '.mat'
        scipy.io.savemat(store_str1,mdict={'Seqs1':Seqs1})
        store_str2 = './validdata_' + str(expInx) + '.mat'
        scipy.io.savemat(store_str2,mdict={'Seqs2':Seqs2})

if __name__ == "__main__":
    main()