import sys
from nltk import tokenize
import random
from gensim.models import keyedvectors
sys.path.append('..')
import utils

class SameWordAugmentor:
    def __init__(self, w2v_path: str, prob: float = 0.1) -> None:
        self.prob = prob
        print('load w2v')
        self.w2v = keyedvectors.load_word2vec_format(w2v_path, limit=1000000, unicode_errors='ignore')
        print('finished')
    
    def augment(self, x: str, x_: str) -> None:
        X, Y = tokenize.word_tokenize(x), tokenize.word_tokenize(x_)
        seq = utils.lcs_seq(X, Y)
        new_y = []
        for ind in seq:
            if random.random() < self.prob and X[ind] in self.w2v:
                cands = self.w2v.most_similar(X[ind])
                cand = random.choice(cands)[0]
                X[ind] = cand
                new_y.append(cand)
            else:
                new_y.append(X[ind])
        return ' '.join(X), ' '.join(new_y)


if __name__ == '__main__':
    aug = SameWordAugmentor(w2v_path='../word2vec/enwiki_20180420_100d.txt', prob=1)
    print(aug.augment('apple is a banana. apple is sweet.', 'apple is sweet.',))
        