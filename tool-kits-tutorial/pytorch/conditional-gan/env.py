import torch
from torch.autograd import Variable
import numpy as np

def artist_works_with_labels(batch_size, paint_points):     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
    paintings = a * np.power(paint_points, 2) + (a-1)
    labels = (a-1) > 0.5            # upper paintings (1), lower paintings (0), two classes
    paintings = torch.from_numpy(paintings).float()
    labels = torch.from_numpy(labels.astype(np.float32))
    return Variable(paintings), Variable(labels)