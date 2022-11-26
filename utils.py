import numpy as np
import torch, string
import random
from nltk import tokenize
from transformers import LogitsProcessor
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]
    """Following steps build L[m+1][n+1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def LCS_score(q_preds, r_preds, q_tgts, r_tgts):
    score = 0
    translator = str.maketrans('', '', string.punctuation)

    for qp, rp, qts, rts in zip(q_preds, r_preds, q_tgts, r_tgts):
        qp, rp = map(lambda x: tokenize.word_tokenize(x.translate(translator)),
                     [qp, rp])
        sub_score = 0
        for qt, rt in zip(qts, rts):
            qt, rt = map(
                lambda x: tokenize.word_tokenize(x.translate(translator)),
                [qt, rt])
            sub_score = max(
                sub_score,
                lcs(qp, qt) / len(set(qp + qt)) +
                lcs(rp, rt) / len(set(rp + rt))
            )
        score += sub_score
    
    return score / (2 * len(q_tgts))


class RestrictWordsProcessor(LogitsProcessor):
    def __init__(self, input_ids: torch.LongTensor, vocab_size: int, ignore_tokens=None) -> None:
        super().__init__()
        onehot = F.one_hot(input_ids.detach().cpu(),
                           num_classes=vocab_size)  # [B, S, V]
        # print(onehot.shape)
        self.restricted_vocab = onehot.sum(1).bool()  # [B, V]
        if ignore_tokens is not None:
            self.restricted_vocab[:, ignore_tokens] = False

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        # print(input_ids.shape)
        # print(scores.shape)
        scores[~self.restricted_vocab] = -float('inf')
        return scores


if __name__ == '__main__':
    q_tgts = [['preds q'], ['preds q2']]
    r_tgts = [['preds r'], ['preds r2']]
    q_preds = ['tgts q']
    r_preds = ['tgts r']
    print(LCS_score(q_preds, r_preds, q_tgts, r_tgts))