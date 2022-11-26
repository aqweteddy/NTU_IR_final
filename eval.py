import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (LogitsProcessorList, AutoModelForSeq2SeqLM,
                          BartForConditionalGeneration)

import utils, csv
from dataset import GenerationTestDataset
from opt import get_parser


def generate(model: AutoModelForSeq2SeqLM, inp_ids, tokenizer, bos_token,
             **kwargs):
    result = model.generate(
        inp_ids,
        max_new_tokens=100,
        early_stopping=True,
        logits_processor=LogitsProcessorList(
            [utils.RestrictWordsProcessor(inp_ids, len(tokenizer))]),
        renormalize_logits=True,
        # forced_bos_token_id=bos_token,
        forced_decoder_ids=[[1, bos_token]],
        num_beams=1,
        **kwargs)
    return tokenizer.batch_decode(result[:, 2:], skip_special_tokens=True)
    # print(result[-1])
    # return tokenizer.batch_decode(result, skip_special_tokens=True)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--dest', default='output.csv', type=str)
    args = parser.parse_args()
    utils.set_seed(args.seed)

    test_df = pd.read_csv(args.data)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained).to('cuda')
    model.eval()
    ds = GenerationTestDataset(pd.read_csv(args.data), args.pretrained,
                               args.template_inp)
    tokenizer = ds.tokenizer
    dl = DataLoader(ds, batch_size=args.batch_size)
    r_bos = tokenizer.encode('[r]', add_special_tokens=False)[0]
    q_bos = tokenizer.encode('[q]', add_special_tokens=False)[0]
    print(r_bos, q_bos)
    r_preds, q_preds, ids = [], [], []
    for idxs, inp_ids, attn_mask in tqdm(dl):
        inp_ids, attn_mask = inp_ids.to('cuda'), attn_mask.to('cuda')
        r_preds.extend(generate(model, inp_ids, tokenizer, r_bos))
        q_preds.extend(generate(model, inp_ids, tokenizer, q_bos))
        ids.extend(idxs.tolist())
        # print(r_preds[-1])
        # print(q_preds[-1])

    result = pd.DataFrame({
        'id': ids,
        'q': [f"\"{q.strip()}\"" for q in q_preds],
        'r': [f"\"{r.strip()}\"" for r in r_preds]
    })
    result.to_csv(args.dest,
                  sep=',',
                  index=False,
                  quoting=csv.QUOTE_NONE,
                  escapechar='\\')
    # with open(args.dest, 'w') as f:
    #     f.write(f'id,q\',r\'\n')
    #     for i, q, r in zip(ids, q_preds, r_preds):
    #         f.write(f'{i},"{q}","{r}"\n')
    # if ground truth exists
    if 'q\'' in test_df.columns:
        r_tgts, q_tgts, idxs = [], [], []
        for idx, grp in test_df.groupby('id'):
            q_tgts.append(grp['q\''].tolist())
            r_tgts.append(grp['r\''].tolist())
            idxs.append(idx)
        print(utils.LCS_score(q_preds, r_preds, q_tgts, r_tgts))
