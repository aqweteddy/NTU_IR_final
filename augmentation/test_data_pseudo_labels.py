import pandas as pd
from argparse import ArgumentParser
import sys
sys.path.append('..')
import utils
import utils, csv
from tqdm import tqdm
from nltk import tokenize


def get_lcs_seq(X, Y):
    X, Y = tokenize.word_tokenize(X), tokenize.word_tokenize(Y)
    seq = utils.lcs_seq(X, Y)
    return ' '.join([X[s] for s in seq])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='data/test.csv')
    parser.add_argument('--pred', default='pseudo_label_base.csv')
    parser.add_argument('--dest', default='data/pseudo_label.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    pred_df = pd.read_csv(args.pred)
    pred_df.columns = ['id', 'q\'', 'r\'']
    df = pd.merge(pred_df, df, on='id')
    print(df.columns)
    q_, r_ = [], []

    for i, row in tqdm(list(df.iterrows())):
        try:
            r_.append(get_lcs_seq(row['r'], row['r\'']))
            q_.append(get_lcs_seq(row['q'], row['q\'']))
        except:
            r_.append(row['r\''])
            q_.append(row['q\''])
    # df.pop('r')
    # df.pop('q')
    # print(df)
    df["q'"], df["r'"] = q_, r_
    df.to_csv(
        args.dest,
        sep=',',
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        #   escapechar='\\'
    )
