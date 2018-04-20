import torch
import matplotlib.pyplot as plt
from brain import Generator, Discriminator
from env import artist_works_with_labels
import numpy as np
from torch.autograd import Variable

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001 # learning rate for generator
LR_D = 0.0001 # learning rate for discriminator
N_IDEAS = 5 # think of this as number of idead for generating an art work (Generators)
ART_COMPONENTS = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])



def train():
  G = Generator(N_IDEAS, ART_COMPONENTS) # Generator
  D = Discriminator(ART_COMPONENTS) # Disriminator
  opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
  opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)

  plt.ion() # something about continuous plotting
  for step in range(10000):
    artist_paitings, labels = artist_works_with_labels(BATCH_SIZE, PAINT_POINTS) # real painting, label from artist
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS)) # random ideas
    G_inputs = torch.cat((G_ideas, labels), 1) # ideas with labels
    G_paintings = G(G_inputs) # fake paiting w.r.t label from G

    D_inputs0 = torch.cat((artist_paitings, labels), 1) # all have their labels
    D_inputs1 = torch.cat((G_paintings, labels), 1)
    prob_artist0 = D(D_inputs0) # D try to increase this prob
    prob_artist1 = D(D_inputs1) # D try to reduce this prob

    D_score0 = torch.log(prob_artist0) # maximize this for D
    D_score1 = torch.log(1 - prob_artist1) # maximize this for D
    D_loss = - torch.mean(D_score0 + D_score1) # minimize the negative of both two above for D
    G_loss = torch.mean(D_score1) # minimize D score w.r.t G

    opt_D.zero_grad()
    D_loss.backward(retain_variables=True) # retain variables for reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 200 == 0:
      plt.cla()
      plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
      bound = [0, 0.5] if labels.data[0, 0] == 0 else [0.5, 1]
      plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound')
      plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound')
      plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(), fontdict={'size': 15})
      plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
      plt.text(-.5, 1.7, 'Class = %i' % int(labels.data[0, 0]), fontdict={'size': 15})
      plt.ylim((0, 3));plt.legend(loc='upper right', fontsize=12);plt.draw();plt.pause(0.1)

  plt.ioff()
  plt.show()
    
def main():
  train()

if __name__ == '__main__':
  main()