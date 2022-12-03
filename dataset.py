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
import torch


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

        self.q: List[str] = df['q'].tolist()
        self.r: List[str] = df['r'].tolist()
        self.s: List[str] = df['s'].tolist()
        self.idx = df['id'].tolist()
        if 'q\'' in df.columns:
            self.q_ = df['q\''].tolist()
            self.r_ = df['r\''].tolist()
        else:
            self.q_ = self.r_ = None

        self.template_inp = template_inp
        # if 't5' in pretrained:
        #     self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
        #         pretrained)
        # else:
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

    def tokenize(self, x: str, maxlen=512, **kwargs):
        dct = self.tokenizer(x,
                             return_tensors='pt',
                             max_length=maxlen,
                             padding='max_length',
                             truncation=True,
                             **kwargs)
        return dct['input_ids'][0], dct['attention_mask'][0]

    @staticmethod
    def swap_template(template: str):
        inp = f'{template}'
        inp = inp.replace("[q] <q>", "<tmp>")
        inp = inp.replace("[r] <r>", "[q] <q>")
        return inp.replace("<tmp>", "[r] <r>")

    @staticmethod
    def get_inputs(q, r, s, template):
        template = template.replace('<q>', q).replace('<r>',
                                                      r).replace('<s>', s)
        template = template.replace("\"", '')
        return template

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

        inp = self.get_inputs(q, r, s, inp)

        inp_ids, attn_mask = self.tokenize(inp, 1024)
        tgt_ids, tgt_attn_mask = self.tokenize(tgt, 200)

        # if tgt_ids[0] == self.tokenizer.bos_token_id:
        #     tgt_ids, tgt_attn_mask = tgt_ids[1:], tgt_attn_mask[1:]
        return {
            'input_ids': inp_ids,
            "attention_mask": attn_mask,
            'labels': tgt_ids
        }


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

        inp_ids, attn_mask = self.tokenize(inp, 1024)
        return self.idx[index], inp_ids, attn_mask


class CopyDataset(GenerationDataset):
    def __init__(self,
                 df: pd.DataFrame,
                 pretrained: str = 'google/t5-v1_1-small',
                 template_inp: str = '[q] is <s> with [r]. [q] <q> [r] <r>',
                 aug_prob: float = 0.):
        super().__init__(df, pretrained, template_inp, aug_prob)

    def __get_tags(self, src: List[str], tgt: List[str]):
        src, tgt = src.tolist(), tgt.tolist()
        idx = len(src) - 1
        while src[idx] == self.tokenizer.pad_token_id:
            idx -= 1
        src = src[:idx + 1]

        idx = len(tgt) - 1
        while tgt[idx] == self.tokenizer.pad_token_id:
            idx -= 1
        tgt = tgt[:idx + 1]
        # src, tgt = self.stripAndSplit(src), self.stripAndSplit(tgt)
        index = self.lcs(src, tgt)
        # tags = [int(i in index) for i in range(len(src))]
        return index

    def __getitem__(self, index):
        q, q_ = self.q[index // 2], self.q_[index // 2]
        r, r_ = self.r[index // 2], self.r_[index // 2]
        s = self.s[index // 2].lower()
        if index % 2 == 0:  # [q]
            inp = f'{self.template_inp}'
            tgt = f'[q] {q_}'
        else:  # [r]
            inp = self.swap_template(self.template_inp)
            tgt = f'[r] {r_}'

        tgt = tgt.replace('"', '')
        inp = inp.replace('"', '')
        tgt_ids, tgt_attn_mask = self.tokenize(tgt, 200)

        inp = self.get_inputs(q, r, s, inp)
        inp_ids, inp_attn_mask = self.tokenize(inp, maxlen=1024)

        tgt_pos = self.__get_tags(inp_ids, tgt_ids)
        tgt_pos += [-100] * (len(tgt_attn_mask) - len(tgt_pos))
        return {
            'input_ids': inp_ids,
            "attention_mask": inp_attn_mask,
            'decoder_input_ids': torch.LongTensor([0] + tgt_ids.tolist()[:-1]),
            'labels': torch.LongTensor(tgt_pos)
        }

    def __len__(self):
        return len(self.q) * 2

    @staticmethod
    def lcs(S1, S2):
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


class RelClsDataset(GenerationDataset):
    def __init__(self,
                 df: pd.DataFrame,
                 pretrained: str = 'google/t5-v1_1-small',
                 aug_prob=0):
        super().__init__(df, pretrained, '', aug_prob)
        self.label2idx = {'DISAGREE': 0, 'AGREE': 1}

    def __len__(self):
        return len(self.q)

    def __getitem__(self, index):
        q = self.q_[index]
        r = self.r_[index]
        s = self.label2idx[self.s[index]]
        if random.random() < self.aug_prob:
            q, r = self.augmetor.augment(q)[0], self.augmetor.augment(r)[0]
        dct = self.tokenizer(q,
                             r,
                             return_tensor='pt',
                             padding='max_length',
                             truncation=True,
                             max_length=250,
                             return_tensors='pt')

        return {
            'input_ids': dct.input_ids[0],
            "attention_mask": dct.attention_mask[0],
            'labels': s
        }


if __name__ == '__main__':
    from tqdm import tqdm
    df = pd.read_csv('data/train.csv')
    ds = CopyDataset(df, 'google/pegasus-x-base', aug_prob=0.)
    # ds = GenerationTestDataset(train_df, 'facebook/bart-base')
    # ds = BIODataset(train_df, 'bert-base-cased', aug_prob=0.5)
    print(ds[0])
    inp_ids, attn_mask, tgt_pos, tgt_attn_mask = ds[1]
    print(ds.tokenizer.decode(inp_ids[tgt_pos]))
    # print(ds[1])
    # dl = data.DataLoader(ds, batch_size=8)
    # maxlen, seq = 0, ""
    # for d in tqdm(ds):
    #     cnt = (d[0] != ds.tokenizer.pad_token_id).sum()
    #     if cnt > maxlen:
    #         maxlen = cnt
    #         seq = d[0]
    # print(maxlen, ds.tokenizer.decode(seq))
    # print(ds[0][2])
    # print(ds[1][2])
    # inp_ids, mask, start, end = ds[0]
    # print(ds.tokenizer.decode(inp_ids[start:end + 1]))
    # for batch in tqdm(ds):
    #     pass
    # print(batch)

    #     break
    # print(train_df.shape, test_df.shape)