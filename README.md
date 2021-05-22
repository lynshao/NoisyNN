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

1. Download the pretrained models first at https://zenodo.org/record/4778688#.YKe8dKgzaUk

2. Enter a folder ('InferenceNoise_SST2' or 'InferenceNoise_CIFAR10');

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

For noisy training, we tried two lightweight NN models (ResNet18 and ShuffleNet V2) on the CIFAR-10 dataset. To run the code, 

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```



## Reproduce our results

The source codes and data to reproduce our results in Fig.3 (noisy inference) are available in the folder 'Reproduce_inference'.

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

The source codes and data to reproduce our results in Fig.4 (noisy training) are available in the folder 'Reproduce_Training'.

Enter the folder and run

```train
python plot_figures.py --arch <shufflenet or resnet18>
```

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
