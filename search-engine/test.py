from search_engine import Bloomfilter, major_segments, segments, Splunk, SplunkX


def test_bloom_filter():
  bf = Bloomfilter(10)
  bf.add_value('dog')
  bf.add_value('cat')
  bf.add_value('fish')
  bf.print_contents()
  bf.add_value('bird')
  bf.print_contents()


  for term in ['dog', 'fish', 'cat', 'bird', 'duck', 'emu']:
    print("{}: {} {}".format(term, bf.hash_value(term), bf.might_contain(term)))

def test_major_segments():
  s = "wo ai ni"
  print(major_segments(s))

def test_segments():
  for term in segments('src_ip = 1.2.3.4'):
    print(term)

def test_splunk():
  s = Splunk()
  s.add_event('src_ip = 1.2.3.4')
  s.add_event('src_ip = 5.6.7.8')
  s.add_event('dst_ip = 1.2.3.4')
  s.add_event('src_ip = 1.2.7.8')

  for event in s.search('1.2.3.4'):
    print(event)
  
  print('-')

  for event in s.search('ip'):
    print(event)

  print('-')
  for event in s.search('src_ip'):
    print(event)

def test_splunkx():
  s = SplunkX()
  s.add_event('src_ip = 1.2.3.4')
  s.add_event('src_ip = 5.6.7.8')
  s.add_event('dst_ip = 1.2.3.4')
  s.add_event('src_ip = 1.2.7.8')

  for event in s.search_all(['src_ip', '5.6.7']):
    print(event)
  print('-')

  for event in s.search_any(['src_ip', 'dst_ip']):
    print(event)
  print('-')
  
def main():
  # test_bloom_filter()
  # test_major_segments()
  # test_segments()
  # test_splunk()
  test_splunkx()

if __name__ == '__main__':
  main()