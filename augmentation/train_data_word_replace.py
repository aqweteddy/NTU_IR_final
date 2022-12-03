import sys
from nltk import tokenize

from .test_data_pseudo_labels import get_lcs_seq
from gensim import word2vec
sys.path.append('..')
import utils

class SameWordAugmentor:
    def __init__(self, w2v_path: str, prob: float = 0.1) -> None:
        self.prob = prob
    
    def augment(self, x: str, x_: str) -> None:
        X, Y = tokenize.word_tokenize(x), tokenize.word_tokenize(x_)
        seq = utils.lcs_seq(X, Y)
        