"""
Created on Mon Dec 15 22:53:55 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn


def check_device_cpu_or_gpu():
    print(f"cuda is avalable: {torch.cuda.is_available()}")
    print(f"cuda device name: {torch.cuda.get_device_name(0)}")
    print(f"cuda version: {torch.version.cuda}")
    print(f"cuda device count: {torch.cuda.device_count()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device} device")
    return device

def tensor_operations():
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(x_data)
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    x_np = torch.tensor(np_array)
    print(x_np)
    tensor = torch.rand(3, 4)
    print(tensor)
    print(f"is accelerator available: {torch.accelerator.is_available()}")
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(tensor)

def get_fashion_datasets():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
    }
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    axs = axs.flatten()
    for i in range(9):
        sample_idx = torch.randint(len(train_data), size=(1, )).item()
        img, label = train_data[sample_idx]
        axs[i].set_title(labels_map[label])
        axs[i].axis("off")
        axs[i].imshow(img.squeeze(), cmap="gray")
    plt.show()
    return train_data, test_data

def load_data(train_data, test_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def run_model():
    device = check_device_cpu_or_gpu()
    model = NeuralNetwork().to(device)
    print(f"model: {model}")
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    print(f"pred_probab: {pred_probab}")
    y_pred = pred_probab.argmax(1)
    print(f"predicted class: {y_pred}")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()
    device = check_device_cpu_or_gpu()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>50}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    test_loss, correct = 0, 0
    device = check_device_cpu_or_gpu()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batch
    correct /= size
    print(f"test error:\naccuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:8f}\n")

def adjust_mode(model, learning_rate, train_dataloader, test_dataloader):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epochs = 10
    for t in range(epochs):
        print(f"eporc {t+1}\n")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("done!")

if __name__ == "__main__":
    print("---- test ----")

    # check_device_cpu_or_gpu()
    # tensor_operations()
    train_data, test_data = get_fashion_datasets()
    train_dataloader, test_dataloader = load_data(train_data, test_data, batch_size=64)
    # run_model()
    device = check_device_cpu_or_gpu()
    model = NeuralNetwork().to(device)
    adjust_mode(model, 1e-3, train_dataloader, test_dataloader)



