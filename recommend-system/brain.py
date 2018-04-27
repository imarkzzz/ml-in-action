from numpy import *
from numpy import linalg as la

def cal_eclud_sim(a, b):
  return 1.0 / (1.0 + la.norm(a - b))

def cal_pears_sim(a, b):
  if len(a) < 3:
    return 1.0
  return 0.5 + 0.5*corrcoef(a, b, rowvar=0)[0][1]

def cal_cos_sim(a, b):
  ab = a.T*b
  denorm = la.norm(a)*la.norm(b)
  return 0.5 + 0.5*(ab/denorm)

