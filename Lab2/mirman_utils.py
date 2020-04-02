import random 
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt 


class LensParser(object):
    def __init__(self):
        pass

    def parse_sequence(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        data = list(filter(lambda l: l.startswith('I') or l.startswith('T'), lines))
        data = list(map(lambda l: eval(l.split('\n')[0][-1]), data))
        return data[0::2]
    
    def parse_labels(self, file, block_size=4, num_labels=5):
        with open(file, 'r') as f:
            lines = f.readlines()
        data = list(filter(lambda l: l.startswith('I') or l.startswith('T'), lines))
        data = list(map(lambda l: eval(l.split(' ')[-1]), data))
        return [(data[k*block_size], *data[block_size*k+2 : block_size*k+4]) for k in range(num_labels)]


class Binarizer(object):
    def __init__(self, vocab_size=20):
        self.vocab_size = vocab_size
        self.encoder = LabelBinarizer()
        self.encoder.fit(list(range(vocab_size)))

    def binarize_sequence(self, seq):
        vectors = self.encoder.transform(seq)

        X = vectors[np.newaxis, :]
        y = np.append(vectors[1:], vectors[:1], axis=0)[np.newaxis, :]
        return X, y 

    def binarize_labels(self, data):
        data_size = len(data)
        word_len = len(data[0]) - 1 
        labels = list(map(lambda l: [l,l], [d[-1] for d in data]))
        data = [d[:-1] for d in data]

        flat_labels = [syllable for label in labels for syllable in label]
        flat_data = [syllable for word in data for syllable in word]

        X = self.encoder.transform(flat_data)[np.newaxis, :]
        y = self.encoder.transform(flat_labels)[np.newaxis, :]
        return X, y 

def multi_plot(losses, accus, mses):
    plt.title('Label Learning Loss Results')
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy loss')
    plt.plot(losses[0], label='W')
    plt.plot(losses[1], label='PW')
    plt.plot(losses[2], label='NW')
    plt.plot(losses[3], label='NWc')
    plt.legend()
    plt.show()

    plt.title('Label Learning Categorical Accuracy Results')
    plt.xlabel('epochs')
    plt.ylabel('categorical accuracy')
    plt.plot(accus[0], label='W')
    plt.plot(accus[1], label='PW')
    plt.plot(accus[2], label='NW')
    plt.plot(accus[3], label='NWc')
    plt.legend()
    plt.show()

    plt.title('Label Learning MSE Results')
    plt.xlabel('epochs')
    plt.ylabel('mean-squarred-error')
    plt.plot(mses[0], label='W')
    plt.plot(mses[1], label='PW')
    plt.plot(mses[2], label='NW')
    plt.plot(mses[3], label='NWc')
    plt.legend()
    plt.show()
