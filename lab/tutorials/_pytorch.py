"""
Created on Thu Oct 16 19:43:02 2025
@author: honda-shin



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import decode_image

def operation_check():
    x = torch.rand(5, 3)
    print(x)
    x = torch.randint(2, 4, (5, 3))
    print(x)
    y = torch.cuda.is_available()
    print(y)

def download_sample_data():
    tran_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=ToTensor())

def show_sample_data(data, sample_idx=None, classes=None):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()
    if sample_idx is None: sample_idx = torch.randint(len(data), size=(9, )).numpy()
    if not isinstance(sample_idx, np.ndarray): sample_idx = np.array([sample_idx])
    for i in range(len(sample_idx)):
        img, label = data[sample_idx[i]]
        axs[i].imshow(img.squeeze(), cmap='gray')
        if classes is not None: label = classes[label]
        axs[i].set_title(f'{label}')
        axs[i].axis('off')
    plt.show()

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pl.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = Path(self.img_dir) / f'{self.img_labels[idx, 1]}'
        image = decode_image(img_path)
        label = self.img_labels[idx, 1]
        if self.transform: image = self.transform(image)
        if self.target_transform: label = self.target_transform(label)
        return image, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            logger.info(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f'test error: \n accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')


if __name__ == '__main__':
    print('---------- test ----------')
    import config
    save_dir = config.ROOT / 'results' / 'tutorial_models'
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(save_dir / 'app.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagae = False

    # operation_check()

    train_data = datasets.FashionMNIST(root='./data', train=True, download=False, transform=ToTensor())
    test_data = datasets.FashionMNIST(root='./data', train=False, download=False, transform=ToTensor())
    # print(type(test_data))
    # show_sample_data(test_data)

    # batch_size = 64
    # train_dataloader = DataLoader(train_data, batch_size=batch_size)
    # test_dataloader = DataLoader(test_data, batch_size=batch_size)
    # for X, y in test_dataloader:
    #     logger.info(f'shape of X: {X.shape}')
    #     logger.info(f'shape of y: {y.shape}')
    #     break

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    # device = 'cpu'
    # logger.info(f'useing {device} device')

    # model = NeuralNetwork().to(device)
    # logger.info(model)

    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # import time
    # st = time.perf_counter()
    # epochs = 100
    # for t in range(epochs):
    #     logger.info(f'epoch: {t+1}\n----------')
    #     train(train_dataloader, model, loss_fn, optimizer)
    #     test(test_dataloader, model, loss_fn)
    # et = time.perf_counter()
    # logger.info(f'Done!, {et-st} [sec]')
    # # cpu: 23.79 sec
    # # cuda: 9.26 sec
    save_path = save_dir / 'model.pth'
    save_path.parent.mkdir(exist_ok=True)
    # logger.debug(save_path)
    # torch.save(model.state_dict(), save_path)
    # logger.info(f"saved pytorch model state to {save_path}")

    model = NeuralNetwork().to(device)
    model_path = save_path
    model.load_state_dict(torch.load(model_path, weights_only=True))

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model.eval()
    sample_idx = 9002
    x, y = test_data[sample_idx][0], test_data[sample_idx][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'predicted: {predicted}, actual: {actual}')

    show_sample_data(test_data, sample_idx=sample_idx, classes=classes)

