from brain import Net
import torch
from env import x, y
import matplotlib.pyplot as plt

def restore_net():
  net = torch.load('net.pkl')
  prediction = net(x)
  plt.scatter(x.data.numpy(), y.data.numpy())
  plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

def restore_params():
  net = Net(n_feature=1, n_hidden=10, n_output=1)
  net.load_state_dict(torch.load('net_params.pkl'))
  prediction = net(x)
  plt.scatter(x.data.numpy(), y.data.numpy())
  plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

# restore_net()
restore_params()
plt.show()