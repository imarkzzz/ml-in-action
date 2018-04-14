import torch
import torch.nn.functional as F # 激活函数都在这

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

def main():
  net = Net(n_feature=1, n_hidden=10, n_output=1)
  print(net) # net 的结构

if __name__ == '__main__':
  main()
