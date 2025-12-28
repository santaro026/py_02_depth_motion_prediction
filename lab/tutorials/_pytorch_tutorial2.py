#%%
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

print(out)
print(x.grad)
out.backward()
print(out)
print(x.grad)


#%%
import torch
import matplotlib.pyplot as plt

p1 = torch.tensor([1, 2])
p2 = torch.tensor([3, 5])
ps = torch.stack([p1, p2], axis=1)
x = ps[0]
y = ps[1]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x, y)
ax.set(xlim=(-10, 10), ylim=(-10, 10))

a = torch.tensor(2., requires_grad=True)
b = torch.tensor(1., requires_grad=True)
print(f'grad of a, b: {a.grad}, {b.grad}')

print(f'a, b: {a}, {b}')
print(f'grad of a, b: {a.grad}, {b.grad}')

learning_rate = 0.16
step = 100
alpha_coff = 0.2
for i in range(step):
    pred = a*x + b
    alpha = alpha_coff * (i + 1)
    if alpha > 1: alpha = 1
    ax.plot(x, pred.detach(), c='r', alpha=alpha)
    e = torch.mean((y-pred)**2)
    e.backward()
    a = (a - a.grad * learning_rate).detach().requires_grad_()
    b = (b - b.grad * learning_rate).detach().requires_grad_()
    print(f'a, b: {a}, {b}')
    print(f'grad of a, b: {a.grad}, {b.grad}')

#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

step = 1000
t = np.linspace(0, 10, step, endpoint=True)
y = np.sin(2*np.pi*t)
fig, ax = plt.subplots()
ax.plot(t, y, c='b')

_t = torch.tensor(0., requires_grad=True)

learning_rate = 0.01
t_grad = []
for i in range(step):
    y = torch.sin(2*np.pi*_t)
    print(f'y: {y}')
    y.backward()
    print(f'_t.grad: {_t.grad}')
    t_grad.append(_t.grad/2/np.pi)
    _t = (_t + learning_rate).detach().requires_grad_()

ax.plot(t, t_grad, c='r')


#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

step = 1000
t = torch.linspace(0, 10, step, requires_grad=True)
y = torch.sin(t)
y.backward(torch.ones_like(t))
dydt = t.grad

fig, ax = plt.subplots()
ax.plot(t.detach(), y.detach(), c='b')
ax.plot(t.detach(), dydt.detach(), c='r')

#%%
import torch
x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
print(y.data.norm())
#%%
import numpy as np
import torch
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

#%%
#### nueral network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, inpu):
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output

net = Net()
print(net)
print('----')
params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)
# net.zero_grad()
# out.backward(torch.randn(1, 10))
# print(params[0].grad)


output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
print(target)
criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# net.zero_grad()
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)

# loss.backward()
# print('conv1.bias.grad ager backward')
# print(net.conv1.bias.grad)

# learning_rate = 0.1
# for f in net.parameters():
    # f.data.sub_(f.grad.data * learning_rate)


import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
print(net.conv1.bias.grad)


#%%
import torch
import torch.nn as nn

a = torch.zeros((1, 100, 100))
print(a.shape)
print(a)

c = nn.Conv2d(1, 6, 3)
print(c.bias)

b = c(a)
print(b.size())
print(b)


