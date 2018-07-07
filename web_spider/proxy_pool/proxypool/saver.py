from proxypool.db import RedisClient

conn = RedisClient()

def set_proxy(proxy):
  result = conn.add(proxy)
  print(proxy)
  print('录入成功' if result else '录入失败')


def scan():
  print('请输入代理,输入exit退出读入')
  while True:
    proxy = input()
    if proxy == 'exit':
      break
    set_proxy(proxy)


def main():
  scan()

if __name__ == '__main__':
  main()