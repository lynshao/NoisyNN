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

3. Run

/InferenceNoise_SST2:
```train
python Main.py --EsN0dB <snr you want to try>
```
/InferenceNoise_CIFAR10:
```train
python Main.py --EsN0dB <snr you want to try> --arch <NN model you want to try: 'resnet34', 'resnet18', or 'shufflenet'>
```

# Reproduce our results

The source code and data to reproduce our results in Fig.3 are available in the folder 'Reproduce_inference'. Enter the folder and run

```train
python plot_figures.py --arch <NN model you want to try: 'resnet34', 'resnet18', or 'shufflenet'>
e.g.,
python plot_figures.py --arch resnet34

```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:  

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
