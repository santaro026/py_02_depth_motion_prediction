"""
Created on Tue Oct 21 18:28:56 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

def download_data(download, transform=None):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)
    return trainset, testset

def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    print('---- test ----')


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset, testset = download_data(download=False, transform=transform)

    batch_size = 4
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    img = torchvision.utils.make_grid(images, nrow=batch_size//2)
    imgshow(img)




