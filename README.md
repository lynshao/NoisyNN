# Denoising Noisy Neural Networks: A Bayesian Approach with Compensation

This repository is the official implementation of paper [Denoising Noisy Neural Networks: A Bayesian Approach with Compensation].
<!-- (add arXiv link) -->

<!-- >📋  Optional: If you find this repository useful, pls cite as include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

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

## Reproduce our results

1. The source codes and data to reproduce our results in Fig.3 (noisy inference) are available in the folder 'Reproduce_inference'.

Enter the folder and run

```train
python plot_figures.py --arch <NN model you want to try: resnet34, resnet18, shufflenet, or bert>
```
> e.g.,
```train
python plot_figures.py --arch bert
```

<!-- <img src="https://github.com/lynshao/NoisyNN/blob/main/Reproduce_inference/Fig3a.png" width="450" alt="Fig3a"/>
<img src="https://github.com/lynshao/NoisyNN/blob/main/Reproduce_inference/Fig3b.png" width="450" alt="Fig3b"/>
<img src="https://github.com/lynshao/NoisyNN/blob/main/Reproduce_inference/Fig3c.png" width="450" alt="Fig3c"/>
<img src="https://github.com/lynshao/NoisyNN/blob/main/Reproduce_inference/Fig3d.png" width="450" alt="Fig3d"/> -->

2. The source codes and data to reproduce our results in Fig.4 (noisy training) are available in the folder 'Reproduce_Training'.

Enter the folder and run

```train
python plot_figures.py --arch <shufflenet or resnet18>
```

## Acknowledgement

K. Liu. Train CIFAR-10 with PyTorch. Available online: https://github.com/kuangliu/pytorch-cifar, MIT license, 2020.

Hugging Face. A pretrained BERT model with text attack. Available online: https://huggingface.co/textattack/bert-base-uncased-SST-2, 2021.

Y. Jiang. SST-2 sentiment analysis. Available online: https://github.com/YJiangcm/SST-2-sentiment-analysis, MIT license, 2020.

S. Ji. A PyTorch implementation of federated learning. Available online: https://github.com/shaoxiongji/federated-learning, MIT license, 2018.
