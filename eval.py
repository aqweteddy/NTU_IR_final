import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (LogitsProcessorList, T5ForConditionalGeneration)

import utils, csv
from dataset import GenerationDataset
from opt import get_parser


def generate(t5: T5ForConditionalGeneration, inp_ids, tokenizer, bos_token,
             **kwargs):

    result = t5.generate(
        inp_ids,
        max_new_tokens=50,
        early_stopping=True,
        logits_processor=LogitsProcessorList(
            [utils.RestrictWordsProcessor(inp_ids, len(tokenizer))]),
        renormalize_logits=True,
        forced_bos_token_id=bos_token,
        forced_decoder_ids=[[0, bos_token]],
        **kwargs)
    return tokenizer.batch_decode(result[:, 2:], skip_special_tokens=True)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--dest', default='output.csv', type=str)
    args = parser.parse_args()
    utils.set_seed(args.seed)

    test_df = pd.read_csv(args.data)
    t5 = T5ForConditionalGeneration.from_pretrained(args.pretrained).to('cuda')
    t5.eval()
    ds = GenerationDataset(pd.read_csv(args.data), args.pretrained,
                           args.template_inp)
    tokenizer = ds.tokenizer
    dl = DataLoader(ds, batch_size=50)
    r_bos = tokenizer.encode('[r]', add_special_tokens=False)[0]
    q_bos = tokenizer.encode('[q]', add_special_tokens=False)[0]

    r_preds, q_preds, ids = [], [], []
    for idxs, inp_ids, attn_mask in tqdm(dl):
        inp_ids, attn_mask = inp_ids.to('cuda'), attn_mask.to('cuda')
        r_preds.extend(generate(t5, inp_ids, tokenizer, r_bos))
        q_preds.extend(generate(t5, inp_ids, tokenizer, q_bos))
        ids.extend(idxs.tolist())

    result = pd.DataFrame({'id': ids, 'q\'': q_preds, 'r\'': r_preds})
    result.to_csv(args.dest, sep=',', index=False, quoting=csv.QUOTE_ALL)

    # if ground truth exists
    if 'q\'' in test_df.columns:
        r_tgts, q_tgts, idxs = [], [], []
        for idx, grp in test_df.groupby('id'):
            q_tgts.append(grp['q\''].tolist())
            r_tgts.append(grp['r\''].tolist())
            idxs.append(idx)
        print(utils.LCS_score(q_preds, r_preds, q_tgts, r_tgts))
