import os
from sys import platform
import argparse, pdb, random, time, scipy.io

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from models import BertModel
from data_sst2 import DataPrecessForSentence
from f_add_noise_and_decode import Add_noise, estimator

def testModel(model, dataloader):
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    accuracy = 0.0
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
    accuracy /= (len(dataloader.dataset))

    return accuracy

def main(args):    
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


    # ================================================================== Load already-trained NN weights
    target_dir = "./"
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
            oneSeq = np.array([])
            for idx2 in range(len(betavec)):
                startTime = time.time()
                # determine lambda_prime and beta
                args.lamb = lambvec[idx1]
                args.beta = betavec[idx2]
                print("-->-->-->-- Running lambda_prime, beta = ",(args.lamb, args.beta))
                # ML or MMSEpb estimation
                denoisedWeights = estimator(weights, args, noisyWeights, varW, varN, StoreRecover)
                # load denoised weights and compute inference accuracy
                net.load_state_dict(denoisedWeights)
                accuracy = testModel(net, test_loader)
                oneSeq = np.append(oneSeq, accuracy)
                print("Accuracy, Time = ",(accuracy, time.time() - startTime))

            if idx1 == 0:
                Seqs = np.array([oneSeq])
            else:
                Seqs = np.r_[Seqs,np.array([oneSeq])]
        
        # Store the search results into matlab file to plot 3D figures
        # matplotlib is also possible, but with less functionality
        store_str = './data_' + str(expInx) + '.mat'
        scipy.io.savemat(store_str,mdict={'Seqs':Seqs})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch NoisyNN')
    parser.add_argument('--Estimator', default=2, type=int, help='ML or MMSEpb estimator')
    parser.add_argument('--EsN0dB', default=9.0, type=float, help='EsN0 in dB')
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args)