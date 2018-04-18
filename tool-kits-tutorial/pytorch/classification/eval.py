from brain import Net
import torch
from env import x, y
import matplotlib.pyplot as plt

def restore_net():
  # restore entire net
  net = torch.load('net.pkl')
  out = net(x)
  prediction = torch.max(out, 1)[1]
  pred_y = prediction.data.numpy().squeeze()
  target_y = y.data.numpy()

  # plot result
  # plt.subplot(132)
  plt.title('Net2')
  plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap="RdYlGn")
  accuracy = sum(pred_y == target_y) / 200
  plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
  plt.show()

def restore_params():
  # restore entire net
  net = Net(n_feature=2, n_hidden=10, n_output=2)
  net.load_state_dict(torch.load('net_params.pkl'))
  out = net(x)
  prediction = torch.max(out, 1)[1]
  pred_y = prediction.data.numpy().squeeze()
  target_y = y.data.numpy()

  # plot result
  # plt.subplot(132)
  plt.title('Net2')
  plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap="RdYlGn")
  accuracy = sum(pred_y == target_y) / 200
  plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
  plt.show()

restore_net()
restore_params()