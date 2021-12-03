# Denoising Noisy Neural Networks: A Bayesian Approach with Compensation

This repository is the official implementation of paper [Denoising Noisy Neural Networks: A Bayesian Approach with Compensation] (https://arxiv.org/abs/2105.10699).

> If you find this repository useful, please kindly cite as
> 
> @article{ShaoNoisyNN,
> 
> title={Denoising Noisy Neural Networks: A Bayesian Approach with Compensation},
> 
> author={Shao, Yulin and Liew, Soung Chang and Gunduz, Deniz},
> 
> journal={arXiv preprint:2105.10699},
> 
> year={2021}
> 
> }

## Requirements

Experiments were conducted on Python 3.8.5. To install requirements:

```setup
pip install -r requirements.txt
```

## Noisy Inference

For noisy inference, we tried three neural network (NN) architectures (ResNet34, ResNet18, and ShuffleNet V2) on the CIFAR-10 dataset and one NN architecture (BERT) on the SST-2 dataset. To run the code, 

1. Download the pretrained models at https://zenodo.org/record/4778688#.YKe8dKgzaUk

2. Enter a folder ('InferenceNoise_CIFAR10' or 'InferenceNoise_SST2');

3. Run the following

> /InferenceNoise_CIFAR10:
```train
python Main.py --EsN0dB <snr you want to try> --arch <NN model you want to try: resnet34, resnet18, or shufflenet>
```

> /InferenceNoise_SST2:
```train
python Main.py --EsN0dB <snr you want to try>
```


## Noisy Training

For noisy training, we tried two lightweight NN models (ShuffleNet V2 and ResNet18) on the CIFAR-10 dataset. To run the code, 

1. Enter the folder 'TrainingNoise_CIFAR10';

3. Start training (please use at least 2 GPUs)

```train
python Main.py --EsN0dB <snr you want to try> --arch <shufflenet or resnet18>
```


## Acknowledgement

K. Liu. Train CIFAR-10 with PyTorch. Available online: https://github.com/kuangliu/pytorch-cifar, MIT license, 2020.

Hugging Face. A pretrained BERT model with text attack. Available online: https://huggingface.co/textattack/bert-base-uncased-SST-2, 2021.

Y. Jiang. SST-2 sentiment analysis. Available online: https://github.com/YJiangcm/SST-2-sentiment-analysis, MIT license, 2020.

S. Ji. A PyTorch implementation of federated learning. Available online: https://github.com/shaoxiongji/federated-learning, MIT license, 2018.
