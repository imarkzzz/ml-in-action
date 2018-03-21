from gensim.models.word2vec import Word2Vec
 
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
 
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)
model.most_similar(positive=['biggest','small'], negative=['big'], topn=5)
model.most_similar(positive=['ate','speak'], negative=['eat'], topn=5)