import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
    parser.add_argument('--EsN0dB', type=float, default=-20.0, help="SNR")
    parser.add_argument('--Estimator', type=float, default=2, help="1->ML,2->MMSEpb")
    parser.add_argument('--withNoise', type=float, default=1, help="w/ or w/o noise")

    # federated edge learning (FEEL) arguments
    parser.add_argument('--num_users', type=int, default=20, help="number of devices")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of active devices")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--local_bs', type=int, default=1024, help="local-training batch size")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="starting learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # model arguments
    # parser.add_argument('--model', type=str, default='cnn', help='model name')
    # parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    # parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
    #                     help='comma-separated kernel size to use for convolution')
    # parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    # parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    # parser.add_argument('--max_pool', type=str, default='True',
    #                     help="Whether use max pooling rather than strided convolutions")

    # other arguments
    # parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    args = parser.parse_args()
    return args
