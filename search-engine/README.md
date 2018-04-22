# Splunk
* Splunk 代表一个拥有搜索功能的索引集合
* 每一个集合中包含一个布隆过滤器，一个倒排词表（字典），和一个存储所有事件的数组
* 当一个事件被加入到索引的时候，会做以下逻辑
  * 为每一个事件生成一个　unique id, 这里就是序号
  * 对事件进行分词，把每一个词加入到倒排词表，也就是每一个词对应的事件的id映射结构，注意，一个词可能对应于多个事件，所以倒排表的值是一个　Set。倒排表是绝大部分搜索引擎的核心功能。
* 当一个词被搜索的时候，会做以下的逻辑
  * 检查布隆过滤器，如果为假，直接返回
  * 检查词表，如果被搜索单词不在词表中，直接返回
  * 在倒排表中找到所有对应的事件id,然后返回事件的内容