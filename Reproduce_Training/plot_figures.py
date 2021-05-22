import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import argparse

# Input
parser = argparse.ArgumentParser(description='PyTorch NoisyNN')
parser.add_argument('--arch', '-a', metavar='ARCH', default= 'shufflenet')
args = parser.parse_args()

# arch = resnet18, shufflenet

# lambda_prime and beta
if args.arch == 'shufflenet':
    lambvec = np.array([0.995, 1])
    betavec = np.arange(-0.6, 0.81, 0.3)
    loc = [2, 3] # location of ML estimation
    EsN0 = np.arange(-26, -15, 2) # SNR range
    savepath = 'Fig4a.png'
elif args.arch == 'resnet18':
    lambvec = np.array([0.85, 0.9, 0.95, 1])
    betavec = np.array([0])
    loc = [4, 1] # location of ML estimation
    EsN0 = np.array([-30, -25, -20, -14, -11, -9, -7, -5]) # SNR range
    savepath = 'Fig4b.png'

def f_get_data(snr, loc, arch, numdata = 10):
    if args.arch == 'resnet18' and snr == -5:
        numdata = 9
    ML = np.zeros(numdata)
    peak = np.zeros(numdata)
    for data in range(numdata):
        path = './data_' + arch + '/neg' + str(abs(snr)) + '/data_' + str(data+1) + '.mat'
        matdict = scipy.io.loadmat(path)
        Seqs = matdict['Seqs']
        Seqs = np.max(Seqs, 2)
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
ML_lower = ML_means - ML_stds * 1
ML_lower[ML_lower < 10] = 10
ML_upper = ML_means + ML_stds * 1
ML_upper[ML_upper > 88] = 88
plt.fill_between(EsN0, ML_lower, ML_upper, color="k", alpha=0.1)
plt.fill_between(EsN0, peak_means - peak_stds * 1, peak_means + peak_stds * 1, color="b", alpha=0.1)
plt.legend(loc = 'upper left')
plt.xlabel('SNR (dB)')
plt.ylabel('Test accuracy')
plt.grid()
plt.style.use('seaborn-whitegrid')
plt.savefig(savepath, dpi=600)
plt.show()
