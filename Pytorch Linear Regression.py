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



class LinearRegressionModel(nn.Module):
    def __init__(self): 
        super().__init__() 
        self.w = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True) # Attribute of Linear Regression
        self.b = nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True) # Attribute of Linear Regression

    def forward(self, inputs): 
        output = self.w * inputs + self.b
        return output



torch.manual_seed(42)


model_0 = LinearRegressionModel()


with torch.inference_mode():
    output = model_0.forward(X_train)

loss_func = nn.L1Loss()

optimizer = torch.optim.SGD(params = model_0.parameters(), # parameters of target model to optimize
                            lr=0.01)
                            
epoch = 100 

for i in range(0, epoch): 
    model_0.train()
    output = model_0.forward(X_train)
    loss = loss_func(output, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss)


















