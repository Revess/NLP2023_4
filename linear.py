import torch
import torch.nn as nn
from dropout import Dropout

x = torch.tensor([3.0,1.0,3.0,4.0])
y = torch.tensor([5.0, 5.0, 5.0, 5.0])
loss_fn = nn.MSELoss()

class Linear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        weights = torch.rand((size_in, size_out),dtype=torch.float32) #initialize the weights randomly of shape, input by output (you could do it like a normal dist)
        bias = torch.zeros((size_out), dtype=torch.float32) #initialize the bias as zero (common practise) 
        self.weights = nn.Parameter(weights, requires_grad=True)
        self.bias = nn.Parameter(bias)
        self.size_in, self.size_out = size_in, size_out


    def forward(self, x):
        x = x.matmul(self.weights)
        x = x + self.bias
        return x

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.weights = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
        # self.bias = torch.tensor([1.0,1.0,1.0,1.0], requires_grad=True)
        self.dropout = Dropout(0.25)
        self.linear = Linear(4, 4)

    def forward(self, x):
        return self.linear(x)

model = BasicModel()
learning_rate = 1e-6
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
for t in range(100):
    model.zero_grad()
    y_pred = model.forward(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optim.step()