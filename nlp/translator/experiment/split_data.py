from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
 
# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

def main():
    # load dataset
    raw_dataset = load_clean_sentences('data/english-german.pkl')
    
    # reduce dataset size
    n_sentences = 10000
    dataset = raw_dataset[:n_sentences, :]
    # random shuffle
    shuffle(dataset)
    # split into train/test
    train, test = dataset[:9000], dataset[9000:]
    # save
    save_clean_data(dataset, 'data/english-german-both.pkl')
    save_clean_data(train, 'data/english-german-train.pkl')
    save_clean_data(test, 'data/english-german-test.pkl')
    
if __name__ == '__main__':
    main() 
