import torch
from torch import nn
from pathlib import Path


weight = 0.3
bias = 0.9

x = torch.arange(0,1.0,0.02)
y = weight * x + bias



train_split = int(0.8 * len(x)) # 80% of data used for training set, 20% for testing 
X_train, y_train = x[:train_split], y[:train_split]
X_test, y_test = x[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

















