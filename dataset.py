from torch.utils import data
import pandas as pd
from transformers import AutoTokenizer, T5Tokenizer


def split_train_test_df(df: pd.DataFrame, frac=float):
    test_idx = pd.Series(df['q'].unique()).sample(frac=frac)
    test_df = df.loc[df['q'].isin(test_idx)]
    train_df = df.loc[~df['q'].isin(test_idx)]
    return train_df, test_df


class Dataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 pretrained: str = 'google/t5-v1_1-small',
                 template_inp: str = '[q] is <s> with [r]. [q] <q> [r] <r>'):
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
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(pretrained)
        if 'google' in pretrained:
            self.tokenizer.add_tokens(["[q]", "[r]"])
            # self.tokenizer.add_special_tokens()
    def __len__(self):
        return len(self.s) * 2

    def tokenize(self, x: str, maxlen=512):
        dct = self.tokenizer(x,
                             return_tensors='pt',
                             max_length=maxlen,
                             padding='max_length',
                             truncation=True)
        return dct['input_ids'][0], dct['attention_mask'][0]

    def __getitem__(self, index):
        q, q_ = self.q[index // 2], self.q_[index // 2]
        r, r_ = self.r[index // 2], self.r_[index // 2]
        s = self.s[index // 2].lower()
        tgt = f'[q] {q_}' if index % 2 == 0 else f'[r] {r_}'
        tgt = tgt.replace('\"', "")

        inp = self.template_inp.replace('<q>', q).replace('<r>',
                                                          r).replace('<s>', s)
        inp = inp.replace("\"", '')

        inp_ids, attn_mask = self.tokenize(inp, 512)

        tgt_ids, tgt_attn_mask = self.tokenize(tgt, 300)
        return inp_ids, attn_mask, tgt_ids, tgt_attn_mask


class GenerationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, pretrained: str = 'google/t5-v1_1-small', template_inp: str = '[q] is <s> with [r]. [q] <q> [r] <r>'):
        super().__init__(df, pretrained, template_inp)
        self.unique_idx = sorted(list(set(self.idx)))
        
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


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    train_df, test_df = split_train_test_df(df, 0.15)
    ds = Dataset(train_df)
    dl = data.DataLoader(ds, batch_size=8)
    print(ds[0][2])
    print(ds[1][2])
    # for batch in dl:
    #     print(batch)
    #     break
    # print(train_df.shape, test_df.shape)