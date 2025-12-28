"""
Created on Sat Oct 18 17:38:40 2025
@author: santaro
tutorial

"""
#%%
#### tensor
import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
# x_np = torch.tensor(np_array)
print(x_np)

#%%
x_ones = torch.ones_like(x_data)
print(x_ones.dtype, x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand.dtype, x_rand)

#%%
shape = (2, 3, )
rand_tensor = torch.ones(shape)
ones_tensor = torch.rand(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

#%%
tensor = torch.rand(3, 4)
print(tensor.shape, tensor.dtype, tensor.device)

#%%
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor.sum())
print(tensor.sum().item())

#%%
#### dataset
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import os
print(os.getcwd())

train_data = datasets.FashionMNIST(root='../data', train=True, download=False, transform=ToTensor())
test_data = datasets.FashionMNIST(root='../data', train=False, download=False, transform=ToTensor())

#%%
from torch.utils.data import DataLoader
batch_size = 8
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(train_features.shape, train_labels.shape)
print(train_features, train_labels)

#%%
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
ds = datasets.FashionMNIST(
    root='../data',
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    # target_transform=Lambda(lambda y: torch.zeros((1, 10), dtype=torch.float).scatter_(dim=1, index=torch.tensor([[y]]), value=1))
)
# ds_features, ds_labels = next(iter(ds))
# print(ds_features.shape, ds_labels.shape)
# print(ds_features, ds_labels)

batch_size = 1
train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(train_features.shape, train_labels.shape)
print(train_features)
print(train_labels)

#%%
#### build model
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {} devide'.format(device))

#%%
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

model = NeuralNetwork().to(device)
print(model)

#%%
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
# y_pred = pred_probab
print(f'predicted class: {y_pred}')

#%%
input_image = torch.rand(3, 28, 28)
print(input_image.size())
# print(input_image.shape)

#%%
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#%%
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
print(hidden1)
hidden1 = nn.ReLU()(hidden1)
print(hidden1.size())
print(hidden1)

#%%
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
print(logits)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)
print(pred_probab.argmax(1))

#%%
print(f'model structure: {model}\n\n')
for name, param in model.named_parameters():
    print(f'layer: {name} | size: {param.size()} | values: {param[:2]} \n')

#%%
#### automatic differentiation with torch.autograd
import torch
x = torch.ones(5)
y = torch.zeros(3)
w = torch.rand(5, 3, requires_grad=True)
b = torch.rand(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f'gradient function for z = {z}')
print(f'gradient function for z = {z.grad_fn}')
print(f'gradient fucntion for loss = {loss}')
print(f'gradient fucntion for loss = {loss.grad_fn}')

#%%
loss.backward()
print(w.grad)
print(b.grad)
#%%
z = torch.matmul(x, w) + b
print(z.requires_grad)
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

#%%
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print(inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(inp.grad)

#%%
inp = torch.ones(8, requires_grad=True)
# inp = torch.ones(8)
out = inp + 1 * inp.sum()
print(f'int: {inp}')
print(f'out: {out}')
out.backward(torch.ones_like(inp), retain_graph=True)
# out.backward(torch.ones_like(inp))
print(f'grad1: {inp.grad}')
out.backward(torch.ones_like(inp), retain_graph=True)
# out.backward(torch.ones_like(inp))
print(f'grad2: {inp.grad}')
out.backward(torch.ones_like(inp), retain_graph=True)
# out.backward(torch.ones_like(inp))
print(f'grad3: {inp.grad}')

#%%
#### optimization
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
train_data = datasets.FashionMNIST(root='../data', train=True, download=False, transform=ToTensor())
test_data = datasets.FashionMNIST(root='../data', train=False, download=False, transform=ToTensor())

batch_size = 64
print(batch_size)
learning_rate = 1e-3
epochs = 10

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
print(test_dataloader)
# print(dir(test_dataloader))
# print(test_dataloader.dataset)

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using {} devide'.format(device))
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
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
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'test error:\n accuracy: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')

for t in range(epochs):
    print(f'epoch {t+1}\n--------------------')
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print(f'done!')

