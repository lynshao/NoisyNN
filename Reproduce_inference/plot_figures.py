import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import argparse

# Input
parser = argparse.ArgumentParser(description='PyTorch NoisyNN')
parser.add_argument('--arch', '-a', metavar='ARCH', default= 'resnet34')
args = parser.parse_args()

# arch = resnet34, resnet18, shufflenet, bert

# lambda_prime and beta
if args.arch == 'resnet34':
    lambvec = np.arange(0.95, 1.071, 0.01)
    betavec = np.arange(-4, 0.51, 0.2)
    loc = [6, 21] # location of ML estimation
    EsN0 = np.arange(-10, -5) # SNR range
    savepath = 'Fig3a.png'
elif args.arch == 'resnet18':
    lambvec = np.arange(0.95, 1.071, 0.01)
    betavec = np.arange(-4, 0.51, 0.2)
    loc = [6, 21] # location of ML estimation
    EsN0 = np.arange(-8, -3) # SNR range
    savepath = 'Fig3b.png'
elif args.arch == 'shufflenet':
    lambvec = np.arange(0.9, 1.151, 0.01)
    betavec = np.arange(-4, 4.1, 0.5)
    loc = [11, 9] # location of ML estimation
    EsN0 = np.arange(-4, 1) # SNR range
    savepath = 'Fig3c.png'
elif args.arch == 'bert':
    lambvec = np.arange(0.1, 1.11, 0.05)
    lambvec = np.concatenate([[0.02, 0.04, 0.06, 0.08], lambvec])
    betavec = np.arange(-10, 51, 2)
    betavec = np.concatenate([[-30, -20, -15], betavec])
    loc = [23, 9] # location of ML estimation
    EsN0 = np.arange(8, 10.1, 0.5) # SNR range
    savepath = 'Fig3d.png'

def f_get_data(snr, loc, arch):
    ML = np.zeros(10)
    peak = np.zeros(10)
    for data in range(10):
        if arch == 'bert':
            path = './data_' + arch + '/' + str(abs(snr)) + '/data_' + str(data+1) + '.mat'
        else:
            path = './data_' + arch + '/neg' + str(abs(snr)) + '/data_' + str(data+1) + '.mat'
        matdict = scipy.io.loadmat(path)
        Seqs = matdict['Seqs']
        ML[data] = Seqs[loc[0]-1,loc[1]-1]
        peak[data] = np.max(Seqs)

    return np.mean(ML), np.std(ML), np.mean(peak), np.std(peak);

ML_means = np.zeros(len(EsN0))
ML_stds = np.zeros(len(EsN0))
peak_means = np.zeros(len(EsN0))
peak_stds = np.zeros(len(EsN0))
for idx in range(len(EsN0)):
    ML_mean, ML_std, peak_mean, peak_std = f_get_data(EsN0[idx], loc, args.arch)
    ML_means[idx] = ML_mean
    ML_stds[idx] = ML_std
    peak_means[idx] = peak_mean
    peak_stds[idx] = peak_std

# plot
plt.figure()
plt.plot(EsN0, ML_means, 'k-', label = 'ML')
plt.plot(EsN0, peak_means, 'b-', label = r'MMSE$_{pb}$')
plt.fill_between(EsN0, ML_means - ML_stds * 1, ML_means + ML_stds * 1, color="k", alpha=0.1)
plt.fill_between(EsN0, peak_means - peak_stds * 1, peak_means + peak_stds * 1, color="b", alpha=0.1)
plt.legend(loc = 'lower right')
plt.xlabel('SNR (dB)')
plt.ylabel('Test accuracy')
plt.grid()
plt.style.use('seaborn-whitegrid')
plt.savefig(savepath, dpi=600)
plt.show()
