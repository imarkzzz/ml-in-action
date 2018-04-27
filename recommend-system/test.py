from unittest import TestCase
from env import *
from brain import *


def main():
  dish_data = load_restaurant_dish()
  dish_data = mat(dish_data)
  cos_sim = cal_cos_sim(dish_data[:, 0], dish_data[:, 4])
  print(cos_sim)

if __name__ == '__main__':
  main()