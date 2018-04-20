import torch
import torch.nn as nn

def Generator(n_ideas, art_compenents):
  G = nn.Sequential(                      # Generator
    nn.Linear(n_ideas+1, 128),          # random ideas (could from normal distribution) + class label
    nn.ReLU(),
    nn.Linear(128, art_compenents),     # making a painting from these random ideas
  )
  return G

def Discriminator(art_compenents): # Discriminator
  D = nn.Sequential(
    nn.Linear(art_compenents+1, 128), # receive artwork either from the famous artist or a newbie like G with label
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(), # tell the probability that the art work is made by artist
  )
  return D