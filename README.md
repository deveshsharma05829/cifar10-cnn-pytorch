# CIFAR-10 Image Classification with a CNN (PyTorch)

A convolutional neural network built from scratch in PyTorch to classify images from the CIFAR-10 dataset. The model reaches 74.54% test accuracy after 10 epochs of training.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 color images (32x32) in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split into 50,000 training and 10,000 test images.

Preprocessing: ToTensor() + Normalize(mean=0.5, std=0.5) per channel.

## Model Architecture

Convolutional layers

| Layer | Details |
|-------|---------|
| Conv2d | 3 → 32, kernel 3, padding 1 + ReLU + MaxPool(2, 2) |
| Conv2d | 32 → 64, kernel 3, padding 1 + ReLU + MaxPool(2, 2) |
| Conv2d | 64 → 128, kernel 3, padding 1 + ReLU + MaxPool(2, 2) |

Fully connected layers

| Layer | Details |
|-------|---------|
| Linear | 4×4×128 → 256 + ReLU |
| Linear | 256 → 10 (class scores) |

## Training Setup

- Loss: CrossEntropyLoss
- Optimizer: Adam (default learning rate)
- Batch size: 64
- Epochs: 10

## Results

| Epoch | Training Loss |
|-------|---------------|
| 1 | 1.383 |
| 5 | 0.524 |
| 10 | 0.174 |

Test accuracy: 74.54%

Training loss keeps dropping while test accuracy plateaus in the mid-70s, which suggests the model is overfitting.

## Getting Started

git clone https://github.com/<your-username>/cifar10-cnn-pytorch.git
cd cifar10-cnn-pytorch
pip install torch torchvision jupyterlab
jupyter labOpen CNN_for_CIFAR10.ipynb and run all cells. The dataset downloads automatically to ./data.

## Future Improvements

- Data augmentation (random crop, horizontal flip)
- Dropout and Batch Normalization
- Learning rate scheduling
- Deeper architecture / residual connections
- Per-class accuracy and confusion matrix

## Tech Stack

Python, PyTorch, torchvision, JupyterLab
