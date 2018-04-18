import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
  def __init__(self, n_feature, n_hidden, n_output):
    super(Net, self).__init__()
    self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
    self.out = torch.nn.Linear(n_hidden, n_output) # output layer

  def forward(self, x):
    x = F.relu(self.hidden(x)) # activation function for hidden layer
    x = self.out(x)
    return x