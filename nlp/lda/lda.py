# -*- coding: utf-8 -*-


import os
import numpy as np
import logging
from collections import defaultdict

# 全局变量
MAX_ITER_NUM = 10000 # 最大迭代次数
VAR_NUM = 20 # 自动计算迭代次数时，计算方差的区间大小


class BiDictionary(object):
  """
  定义双向字典，通过key可以得到value,通过value也可以得到key
  """

  def __init__(self):
    """
    :key: 双向字典初始化
    """
    self.dict = {} # 正向的数据字典，其 key 为 self 的 key
    self.dict_reverssed = {} # 反向的数据字典，其 key 为 self 的 value
    return

  def __str__(self):
    """
    :key: 将双向字典转化为字符串对象
    """
    str_list = ["%s\t%s" % (key, self.dict[key]) for key in self.dict]
    return "\n".join(str_list)
  
  def clear(self):
    """
    :key: 清空双向字典对象
    """
    self.dict.clear()
    self.dict_reverssed.clear()
    return
  
  def add_key_value(self, key, value):
    """
    :key: 更新双向字典，增加一项
    """
    self.dict[key] = value
    self.dict_reverssed[value] = key
    return

  def remove_key_value(self, key, value):
    """
    :key: 更新双向字典，删除一项
    """
    if key in self.dict:
      del self.dict[key]
      del self.dict_reverssed[value]
    return

  def get_value(self, key, default=None):
    """
    :key: 通过key获取value，不存在返回default
    """
    return self.dict.get(key, default)
  
  def get_key(self, value, default=None):
    """
    :key: 通过value获取key，不存在返回default
    """
    return self.dict_reverssed.get(key, default)
  
  def contains_key(self, key):
    """
    :key: 判断是否存在key值
    """
    return key in self.dict
  
  def contains_value(self, value):
    """
    :key: 判断是否存在value值
    """
    return value in self.dict_reverssed
  
  def keys(self):
    """
    :key: 得到双向字典全部的keys
    """
    return self.dict.keys()

  def values(self):
    """
    :key: 得到双向字典全部的values
    """
    return self.dict_reverssed.keys()
  
  def items(self):
    """
    :key: 得到双向字典全部的items
    """
    return self.dict.items()
  

class CorpusSet(object):
  """
  定义语料集类，作为LdaBase的基类
  """
  
  def __init__(self):
    """
    :key: 初始化函数
    """
    # 定义关于word的变量
    self.local_bi = BiDictionary() # id和word之间的本地双向字典,key为id,value为word
    self.words_count = 0 # 数据集中word的数量（排重之前的）
    self.V = 0 # 数据集中word的数量（排重之后的）

    # 定义关于article的变量
    self.artids_list = [] # 全部article的id的列表，按照数据读取的顺序存储
    self.arts_Z = [] # 全部article中所有词的id信息，维数为 M*art.length()
    self.M = 0 # 数据集中article的数量

    # 定义推断中用到的变量（可能为空）
    self.global_bi = None # id和word之间的全局双向字典，key为id,value为word
    self.local_2_global = {} # 一个字典,local字典和global字典之间的对应关系
    return

  def init_corpus_with_file(self, file_name):
    """
    利用数据文件初始化语料集数据。文件的每一行的数据格式：
    id\tword1 word2 word3...
    """
    with open(file_name, "r", encoding="utf-8") as file_iter:
      self.init_corpus_with_articles(file_iter)
    return
  
  def init_corpus_with_articles(self, article_list):
    """
    利用article的列表初始化语料集。每一篇article的格式为
    id\tword1 word2 word3...
    """
    # 准备数据--word数据
    self.local_bi.clear()
    self.words_count = 0
    self.V = 0

    # 清理数据--article数据
    self.artids_list.clear()
    self.arts_Z.clear()
    self.M = 0

    # 清理数据--清理local到global的映射关系
    self.local_2_global.clear()

    # 读取article数据
    for line in article_list:
      frags = line.strip().split()
      if len(frags) < 2:
        continue

      # 获取article的id
      art_id = frags[0].strip()

      # 获取word的id
      art_wordid_list = []
      for word in [w.strip() for w in frags[1:] if w.strip()]:
        local_id = self.local_bi.get_key(word) if self.contains_value(word) else len(self.local_bi)

        # 这里的self.global_bi为None和为空是有区别的
        if self.global_bi is None:
          # 更新id信息
          self.local_bi.add_key_value(local_id, word)
          art_wordid_list.append(local_id)
        else:
          if self.global_bi.contains_value(word):
            # 更新id信息
            self.local_bi.add_key_value(local_id, word)
            art_wordid_list.append(local_id)

            # 更新local_2_global
            self.local_2_global[local_id] = self.global_bi.get_key(word)
      
      # 更新类变量：必须article中word的数量大于0
      if len(art_wordid_list) > 0:
        self.words_count += len(art_wordid_list)
        self.artids_list.append(art_id)
        self.arts_Z.append(art_wordid_list)

    # 做相关处室计算--word相关
    self.V = len(self.local_bi)
    logging.debug("words number: " + str(self.V) + ", " + str(self.words_count))

    # 做相关初始计算--article相关
    self.M = len(self.artids_list)
    logging.debug("articles number: " + str(self.M))
    return

  def save_wordmap(self, file_name):
    """
    保存word字典，即self.local_bi的数据
    """
    with open(file_name, "w", encoding="utf-8") as f_save:
      f_save.write(str(self.local_bi))
    return

  def load_word_map(self, file_name):
    """
    加载word字典，即加载self.local_bi的数据
    """
    self.local_bi.clear()
    with open(file_name, "r", encoding="utf-8") as f_load:
      for _id, _word in [line.strip().split() for line in f_load if line.strip()]:
        self.local_bi.add_key_value(int(_id), _word.strip())
    self.V = len(self.local_bi)
    return
