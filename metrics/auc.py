# -*- coding: utf-8 -*-
# 　auc值的大小可以理解为: 随机抽一个正样本和一个负样本，正样本预测值比负样本大的概率
import random
import time


def timeit(func):
  """
  装饰器，计算函数执行时间
  """

  def wrapper(*args, **kwargs):
    time_start = time.time()
    result = func(*args, **kwargs)
    time_end = time.time()
    exec_time = time_end - time_start
    print("{function} exec time: {time}s".format(function=func.__name__, time=exec_time))
    return result
  
  return wrapper


def gen_label_pred(n_sample):
  """
  随机生成n个样本的标签和预测值
  """
  labels = [random.randint(0, 1) for _ in range(n_sample)]
  preds = [random.random() for _ in range(n_sample)]
  return labels, preds


@timeit
def naive_auc(labels, preds):
  """
  最简单粗暴的方法
  先排序，然后统计有多少正负样本对满足：正样本预测值>负样本预测值，再除以总的正负样本对个数
  复杂度 O(NlogN)，N为样本数
  """
  n_pos = sum(labels)
  n_neg = len(labels) - n_pos
  total_pair = n_pos * n_neg

  labels_preds = zip(labels, preds)
  labels_preds = sorted(labels_preds, key=lambda x: x[1])
  accumulated_neg = 0
  satisfied_pair = 0
  for i in range(len(labels_preds)):
    if labels_preds[i][0] == 1:
      satisfied_pair += accumulated_neg
    else:
      accumulated_neg += 1
  
  return satisfied_pair / float(total_pair)


@timeit
def approximate_auc(labels, preds, n_bins=100):
  """
  近似方法，将预测值分桶(n_bins),对正负样本分别构建直方图，再统计满足条件的正负样本对
  复杂度 O(N)
  这种方法有什么缺点？怎么分桶？
  """
  n_pos = sum(labels)
  n_neg = len(labels) - n_pos
  total_pair = n_pos * n_neg

  pos_histogram = [0 for _ in range(n_bins)]
  neg_histogram = [0 for _ in range(n_bins)]
  bin_width = 1.0 / n_bins
  for i in range(len(labels)):
    nth_bin = int(preds[i] / bin_width)
    if labels[i] == 1:
      pos_histogram[nth_bin] += 1
    else:
      neg_histogram[nth_bin] += 1
  
  accumulated_neg = 0
  satisfied_pair = 0
  for i in range(n_bins):
    satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
    accumulated_neg += neg_histogram[i]
  
  return satisfied_pair / float(total_pair)

# 思考: mapreduce 版本的auc怎么写


if __name__ == "__main__":
  labels, preds = gen_label_pred(100000)
  naive_auc_rs = naive_auc(labels, preds)
  approximate_auc_rs = approximate_auc(labels, preds)
  print("naive auc result:{}, approximate auc result:{}".format(naive_auc_rs, approximate_auc_rs))

