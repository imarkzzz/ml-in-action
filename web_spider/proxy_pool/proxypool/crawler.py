import json
import re
from pyquery import PyQuery as pq


class ProxyMetaclass(type):
  def __new__(cls, name, bases, attrs):
    count = 0
    attrs['__CrawlFunc__'] = []
    for k, v in attrs.items():
      if 'crawl_' in k:
        attrs['__CrawlFunc__'].append(k)
        count += 1
    attrs['__CrawlFuncCount__'] = count
    return type.__new__(cls, name, bases, attrs)


class Crawler(object, metaclass=ProxyMetaclass):
  def get_proxies(self, callback):
    proxies = []
    for proxy in eval("self.{}".format(callback)):
      print("成功获取到代理", proxy)
    return proxies
    