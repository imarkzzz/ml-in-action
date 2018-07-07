import json
import re
from .utils import get_page
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
  
  def crawl_daili66(self, page_count=4):
    """
    获取代理66
    :param page_count: 页码
    :return: 代理
    """
    start_url = 'http://www.66ip.cn/{}.html'
    urls = [start_url.format(page) for page in range(1, page_count + 1)]
    for url in urls:
      print('Crawling', url)
      html = get_page(url)
      if html:
        doc = pq(html)
        trs = doc('.containerbox table tr:gt(0)').items()
        for tr in trs:
          ip = tr.find('td:nth-child(1)').text()
          port = tf.find('td:nth-child(2)').text()
          yield ':'.join([ip, port])
  

  def crawl_proxy360(self):
    """
    获取Proxy360
    :return: 代理
    """
    start_url = 'http://www.proxy360.cn/Region/China'
    print('Crawling', start_url)
    html = get_page(start_url)
    if html:
      doc = pq(html)
      lines = doc('div[name="list_proxy_ip"]').items()
      for line in lines:
        ip = line.find('.tbBottomLine:nth-child(1)').text()
        port = line.find('.tbBottomLine:nth-child(2)').text()
        yield ':'.join([ip, port])

  
  def crawl_goubanjia(self):
    """"
    获取Goubanjia
    :return: 代理
    """
    start_url = 'http://www.goubanjia.com/tree/gngn/index.shtml'
    html = get_page(start_url)
    if html:
      doc = pq(html)
      tds = doc('td.ip').items()
      for td in tds:
        td.find('p').remove()
        yield td.text().replace(' ', '')


  def crawl__ip181(self):
    """
    获取Goubanjia
    :return: 代理
    """
    start_url = 'http://www.goubanjia.com/free/gngn/index.shtml'
    html = get_page(start_url)
    if html:
      doc = pq(html)
      tds = doc('td.ip').items()
      for td in tds:
        td.find('p').remove()
        yield td.text().replace(' ', '')

  
  def crawl_ip3366(self):
    for page in range(1, 4):
      start_url = 'http://www.ip3366.net/free/?stype=1&page={}'.format(page)
      html = get_page(start_url)
      ip_address = re.compile('<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>')
      # \s * 匹配空格,起到换行作用
      re_ip_address = ip_address.findall(html)
      for address, port in re_ip_address:
        result = address + ':' + port
        yield result.replace(' ', '')

  
  def crawl_kxdaili(self):
    for i in range(1, 11):
      start_url = 'http://www.kxdaili.com/ipList/{}.html#ip'.format(i)
      html = get_page(start_url)
      ip_address = re.compile('<tr.*?>\s*<td>(.*?)</td>\s*<td>(.*?)</td>')
      # \s * 匹配空格,起到换行作用
      re_ip_address = ip_address.findall(html)
      for address, port in re_ip_address:
        result = address + ':' + port
        yield result.replace(' ', '')
  

  def crawl_premproxy(self):
    for i in ['China-01', 'China-02', 'China-03', 'China-04', 'Taiwan-01']:
      start_url = 'https://premproxy.com/proxy-by-country/{}.htm'.format(i)
      html = get_page(start_url)
      if html:
        ip_address = re.compile('<td data-label="IP:port ">(.*?)</td>')
        re_ip_address = ip_address.findall(html)
        for address_port in re_ip_address:
          yield address_port.replace(' ', '')