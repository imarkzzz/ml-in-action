from brain import Net
import torch
import matplotlib.pyplot as plt
from env import x, y

def train():
  net = Net(n_feature=2, n_hidden=10, n_output=2) # define the network
  print(net) # net architecture
  
  optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
  loss_func = torch.nn.CrossEntropyLoss() # the target label is NOT an one-hotted
  
  plt.ion() # something about plotting
  
  for t in range(100):
    out = net(x) # input x and predict based on x
    loss = loss_func(out, y) # must be (1. nn output, 2.targer), the target label is NOT one-hotted
    optimizer.zero_grad() # clear gradients for next train
    loss.backward() # backpropagation, compute gradients
    optimizer.step() # apply gradients
  
    if t % 2 == 0:
      # plot and show learning process
      plt.cla()
      prediction = torch.max(out, 1)[1]
      pred_y = prediction.data.numpy().squeeze()
      target_y = y.data.numpy()
      plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap="RdYlGn")
      accuracy = sum(pred_y == target_y) / 200
      plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
      plt.pause(0.1)
  
  torch.save(net, 'net.pkl') # entire net
  torch.save(net.state_dict(), 'net_params.pkl')
  plt.ioff()
  plt.show()