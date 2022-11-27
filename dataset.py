from torch.utils import data
import pandas as pd
from transformers import AutoTokenizer, T5Tokenizer
from typing import List
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import nlpaug.flow as naf
from nlpaug.util import Action
import random
import utils


def split_train_test_df(df: pd.DataFrame, frac=float):
    test_idx = pd.Series(df['q'].unique()).sample(frac=frac)
    test_df = df.loc[df['q'].isin(test_idx)]
    train_df = df.loc[~df['q'].isin(test_idx)]
    return train_df, test_df


class GenerationDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 pretrained: str = 'google/t5-v1_1-small',
                 template_inp: str = '[q] is <s> with [r]. [q] <q> [r] <r>',
                 aug_prob=0):
        if '<q>' not in template_inp or '<r>' not in template_inp or '<s>' not in template_inp:
            raise ValueError('template error')

        self.q = df['q'].tolist()
        self.r = df['r'].tolist()
        self.s = df['s'].tolist()
        self.idx = df['id'].tolist()
        if 'q\'' in df.columns:
            self.q_ = df['q\''].tolist()
            self.r_ = df['r\''].tolist()
        else:
            self.q_ = self.r_ = None

        self.template_inp = template_inp
        if 't5' in pretrained:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                pretrained)
        else:
            self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                pretrained)

        if "[q]" not in self.tokenizer.get_added_vocab():
            self.tokenizer.add_tokens(["[q]", "[r]"])
            print("token added")
            # self.tokenizer.add_special_tokens()

        self.aug_prob = aug_prob
        self.augmetor = naf.Sometimes([
            naw.RandomWordAug(action='swap'),
            # naw.RandomWordAug(action='crop'),
            # naw.AntonymAug(),
            naw.RandomWordAug(),
        ])

    def __len__(self):
        return len(self.s) * 2

    def tokenize(self, x: str, maxlen=512):
        dct = self.tokenizer(x,
                             return_tensors='pt',
                             max_length=maxlen,
                             padding='max_length',
                             truncation=True)
        return dct['input_ids'][0], dct['attention_mask'][0]
    
    @staticmethod
    def swap_template(template: str):
        inp = f'{template}'
        inp = inp.replace("[q] <q>", "<tmp>")
        inp = inp.replace("[r] <r>", "[q] <q>")
        return  inp.replace("<tmp>", "[r] <r>")
    
    def __getitem__(self, index):
        q, q_ = self.q[index // 2], self.q_[index // 2]
        r, r_ = self.r[index // 2], self.r_[index // 2]
        s = self.s[index // 2].lower()

        if random.random() < self.aug_prob:
            q, r = self.augmetor.augment(q)[0], self.augmetor.augment(r)[0]

        tgt = f'[q] {q_}' if index % 2 == 0 else f'[r] {r_}'
        tgt = tgt.replace('\"', "")
        if index % 2 == 1:  # tgt [r]
            inp = self.swap_template(self.template_inp)
        else:
            inp = f'{self.template_inp}'
            
        inp = inp.replace('<q>', q).replace('<r>', r).replace('<s>', s)
        inp = inp.replace("\"", '')
        inp_ids, attn_mask = self.tokenize(inp, 512)

        tgt_ids, tgt_attn_mask = self.tokenize(tgt, 250)

        if tgt_ids[0] == self.tokenizer.bos_token_id:
            tgt_ids, tgt_attn_mask = tgt_ids[1:], tgt_attn_mask[1:]

        return inp_ids, attn_mask, tgt_ids, tgt_attn_mask


class GenerationTestDataset(GenerationDataset):
    def __init__(self,
                 df: pd.DataFrame,
                 pretrained: str = 'google/t5-v1_1-small',
                 template_inp: str = '[q] is <s> with [r]. [q] <q> [r] <r>'):
        super().__init__(df, pretrained, template_inp)
        self.unique_idx = sorted(list(set(self.idx)))
        print(self.tokenizer.get_added_vocab())

    def __len__(self):
        return len(self.unique_idx)

    def __getitem__(self, index):
        index = self.idx.index(self.unique_idx[index])

        q = self.q[index]
        r = self.r[index]
        s = self.s[index].lower()

        inp = self.template_inp.replace('<q>', q).replace('<r>',
                                                          r).replace('<s>', s)
        inp = inp.replace("\"", '')

        inp_ids, attn_mask = self.tokenize(inp, 512)
        return self.idx[index], inp_ids, attn_mask


class BIODataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 pretrained: str = 'bert-base-uncased',
                 aug_prob: float = 0.):

        self.q = df['q'].tolist()
        self.r = df['r'].tolist()
        self.s = df['s'].tolist()
        self.idx = df['id'].tolist()
        if 'q\'' in df.columns:
            self.q_ = df['q\''].tolist()
            self.r_ = df['r\''].tolist()
        else:
            self.q_ = self.r_ = None

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self.aug_prob = aug_prob
        self.augmetor = naf.Sometimes([
            naw.RandomWordAug(action='swap'),
            # naw.RandomWordAug(action='crop'),
            # naw.AntonymAug(),
            naw.RandomWordAug(),
        ])

    @staticmethod
    def stripAndSplit(string):
        ret = string
        ret = ret[1:] if ret.startswith("\"") else ret
        ret = ret[:-1] if ret.endswith("\"") else ret
        ret = ret.split()
        return ret

    def tagging(self, q, q_, r, r_):
        qtags = []
        q = self.stripAndSplit(q)
        qp = self.stripAndSplit(q_)

        qindex = self.lcs(q, qp)
        for i in range(len(q)):
            if i in qindex:
                qtags.append(1)
            else:
                qtags.append(0)

        rtags = []
        r = self.stripAndSplit(r)
        rp = self.stripAndSplit(r_)

        rindex = self.lcs(r, rp)
        for i in range(len(r)):
            if i in rindex:
                rtags.append(1)
            else:
                rtags.append(0)

        return qtags, rtags

    def __getitem__(self, index):
        q, q_ = self.q[index], self.q_[index]
        r, r_ = self.r[index], self.r_[index]
        s = self.s[index].lower()
        qlabel, rlabel = self.tagging(q, q_, r, r_)
        inputs = self.tokenizer(q + s,
                                r,
                                is_split_into_words=True,
                                truncation=True,
                                padding='max_length')
        inputs["labels"] = self.align_labels_with_tokens(
            qlabel + [-100] + rlabel, inputs.word_ids())
        return inputs

    def __len__(self):
        return len(self.q)

    @staticmethod
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)
            else:
                label = labels[word_id]
                new_labels.append(label)

        return new_labels

    @staticmethod
    def lcs(X, Y):
        m = len(X)
        n = len(Y)

        L = [[0 for i in range(n + 1)] for j in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        index = []
        i = 0
        j = 0
        while i < m and j < n:
            if X[i] == Y[j]:
                index.append(i)
                i += 1
                j += 1
            else:
                i += 1

        return index[::-1]


if __name__ == '__main__':
    from tqdm import tqdm
    df = pd.read_csv('data/train.csv')
    train_df, test_df = split_train_test_df(df, 0.15)
    ds = GenerationDataset(train_df, 'facebook/bart-base', aug_prob=0.)
    # ds = GenerationTestDataset(train_df, 'facebook/bart-base')
    # ds = BIODataset(train_df, 'bert-base-cased', aug_prob=0.5)
    # print(ds[0])
    # print(ds[1])
    ds[0]
    ds[1]
    ds[2]
    ds[3]
    # dl = data.DataLoader(ds, batch_size=8)

    # print(ds[0][2])
    # print(ds[1][2])
    # inp_ids, mask, start, end = ds[0]
    # print(ds.tokenizer.decode(inp_ids[start:end + 1]))
    for batch in tqdm(dl):
        pass
        # print(batch)

    #     break
    # print(train_df.shape, test_df.shape)