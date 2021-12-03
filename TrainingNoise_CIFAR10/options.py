import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
    parser.add_argument('--EsN0dB', type=float, default=-20.0, help="SNR")
    parser.add_argument('--denoiser', type=float, default=2, help="1->ML,2->MMSEpb")
    parser.add_argument('--withNoise', type=float, default=1, help="w/ or w/o noise")

    # federated edge learning (FEEL) arguments
    parser.add_argument('--num', default=1, type=int)
    parser.add_argument('--num_users', type=int, default=20, help="number of devices")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of active devices")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--local_bs', type=int, default=1024, help="local-training batch size")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="starting learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    args = parser.parse_args()
    return args
