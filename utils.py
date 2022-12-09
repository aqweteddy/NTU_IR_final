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


def get_lcs_seq(X, Y):
    X, Y = tokenize.word_tokenize(X), tokenize.word_tokenize(Y)
    seq = lcs_seq(X, Y)
    return ' '.join([X[s] for s in seq])


def lcs_seq(S1, S2):
    m, n = len(S1), len(S2)
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Building the mtrix in bottom-up way
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif S1[i - 1] == S2[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    index = L[m][n]

    lcs_algo = [""] * (index + 1)
    lcs_algo[index] = ""

    i = m
    j = n
    result = []
    while i > 0 and j > 0:

        if S1[i - 1] == S2[j - 1]:
            lcs_algo[index - 1] = S1[i - 1]
            i -= 1
            j -= 1
            index -= 1
            result.append(i)

        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return result[::-1]


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


def LCS_score_side(preds, tgts):
    score = 0
    translator = str.maketrans('', '', string.punctuation)
    for p, ts in zip(preds, tgts):
        sub_score = 0
        p = tokenize.word_tokenize(p.translate(translator))
        for t in ts:
            t = tokenize.word_tokenize(t.translate(translator))
            lcs_ = lcs(p, t)
            sub_score = max(sub_score, lcs(p, t) / max(len(p) + len(t) - lcs_, 1))
        score += sub_score
    return score / len(tgts)


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
            lcs_q, lcs_r = lcs(qp, qt), lcs(rp, rt)
            sub_score = max(
                sub_score, lcs_q / (len(qp) + len(qt) - lcs_q) + lcs_r /
                (len(rp) + len(rt) - lcs_r))
        score += sub_score

    return score / (2 * len(q_tgts))


class RestrictWordsProcessor(LogitsProcessor):
    def __init__(self,
                 input_ids: torch.LongTensor,
                 vocab_size: int,
                 ignore_tokens=None) -> None:
        super().__init__()
        onehot = F.one_hot(input_ids.detach().cpu(),
                           num_classes=vocab_size)  # [B, S, V]
        # print(onehot.shape)
        self.vocab_size = vocab_size
        self.restricted_vocab = onehot.sum(1).bool()  # [B, V]
        if ignore_tokens is not None:
            self.restricted_vocab[:, ignore_tokens] = False

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        # print(input_ids.shape)
        # print(scores.shape)
        batch_size = scores.shape[0]
        if batch_size != self.restricted_vocab.shape[0]:
            num_seq = int(batch_size / self.restricted_vocab.shape[0])
            mask = (~self.restricted_vocab).repeat(1, num_seq).reshape(
                -1, self.vocab_size)
            scores[mask] = -float('inf')
        else:
            scores[~self.restricted_vocab] = -float('inf')
        return scores


class CopyLogitProcessor(LogitsProcessor):
    def __init__(self,
                 input_ids: torch.LongTensor,
                 vocab_size: int,
                 restrict_direction=False) -> None:
        super().__init__()
        self.src_ids = input_ids  # [B, S]
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: B, 1
        scores: B, S
        -> B, vocab_size
        """
        result = torch.zeros(input_ids.shape[0],
                             self.vocab_size,
                             device=input_ids.device)
        # next_indices = scores.argmax(-1)
        for i in range(len(input_ids)):
            result[i, src_id] = scores

        return result


if __name__ == '__main__':
    q_tgts = [['preds q'], ['preds q2']]
    r_tgts = [['preds r'], ['preds r2']]
    q_preds = ['tgts q']
    r_preds = ['tgts r']
    print(LCS_score(q_preds, r_preds, q_tgts, r_tgts))
    print(LCS_score_side(q_preds, q_tgts))