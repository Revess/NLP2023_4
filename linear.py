import torch
import torch.nn as nn

#
# x = torch.tensor([3.0,1.0,3.0,4.0], requires_grad=True)
# y = torch.tensor([5.0, 5.0, 5.0, 5.0])
# loss_fn = nn.MSELoss()

class Linear(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        weights = torch.tensor(torch.zeros(size_out, size_in),dtype=torch.float32)
        bias = torch.tensor(size_out, dtype=torch.float32)
        self.weights = nn.Parameter(weights, requires_grad=True)
        self.bias = nn.Parameter(bias)
        self.size_in, self.size_out = size_in, size_out


    def forward(self, x):
        x = x * self.weights
        x = x + self.bias
        return x


# class BasicModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.weights = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
#         # self.bias = torch.tensor([1.0,1.0,1.0,1.0], requires_grad=True)
#         self.linear = Linear(4, 1)
#
#     def forward(self, x):
#         return self.linear(x)
#
# model = BasicModel()
# learning_rate = 1e-6
# optim = torch.optim.Adam(model.parameters(), lr=1e-4)
# for t in range(20000):
#     model.zero_grad()
#     if t == 0 or t == 500 or t == 1000 or t == 1500:
#         print(model.linear.weights)
#         print(model.linear.bias)
#     y_pred = model.forward(x)
#     loss = loss_fn(y_pred, y)
#     loss.backward()
#     optim.step()