import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (LogitsProcessorList, AutoModelForSeq2SeqLM,
                          BartForConditionalGeneration)

import utils, csv
from dataset import GenerationTestDataset
from opt import get_parser
from model import LongT5ForCopyGeneration


def generate(model: AutoModelForSeq2SeqLM, inp_ids: torch.Tensor,
             attn_mask: torch.Tensor, tokenizer, bos_token, device='cuda', **kwargs):
    tr_pos = len(inp_ids[0]) - 1
    while (inp_ids[:, tr_pos] != 0).sum() == 0:
        tr_pos -= 1
    tr_pos += 1
    inp_ids, attn_mask = inp_ids[:, :tr_pos], attn_mask[:, :tr_pos]

    r_bos = tokenizer.encode('[r]', add_special_tokens=False)[0]
    q_bos = tokenizer.encode('[q]', add_special_tokens=False)[0]
    vocab = inp_ids.clone().detach()
    q_idx = (vocab == q_bos).nonzero()[1::2]
    r_idx = (vocab == r_bos).nonzero()[1::2]

    if bos_token == "[r]":
        for r in r_idx:
            vocab[r[0], :r[1]] = tokenizer.pad_token_id
    else:  # [q]
        # [q] .. [r]
        for p in q_idx:
            vocab[p[0], :p[1]] = tokenizer.pad_token_id

        for p in r_idx:
            vocab[p[0], p[1]:] = tokenizer.pad_token_id
    bos_token_id = r_bos if bos_token == '[r]' else q_bos

    inp_ids, attn_mask = inp_ids.to(device), attn_mask.to(device)
    beam_result = model.generate(
        inp_ids,
        attention_mask=attn_mask,
        max_new_tokens=150,
        early_stopping=True,
        # logits_processor=LogitsProcessorList(
        #     [utils.RestrictWordsProcessor(inp_ids, len(tokenizer))]),
        renormalize_logits=True,
        # forced_bos_token_id=bos_token_id,
        # decoder_start_token_id=bos_token_id,
        forced_decoder_ids=[[1, bos_token_id]],
        top_p=0.15,
        num_beams=1,
        num_return_sequences=1,
        **kwargs)
    result = beam_result
    # beam_result = beam_result.reshape(-1, 3, beam_result.shape[-1])
    # for preds, tgt_ids in zip(beam_result, inp_ids.tolist()):
    #     max_seq, max_score = [], 0
    #     tgt_ids = [w for w in tgt_ids if w != 0]
    #     # print(tokenizer.batch_decode(preds, skip_special_tokens=True))
    #     for p in preds:
    #         sc = utils.lcs([w for w in p if w != 0], tgt_ids) / (len(tgt_ids) + len(p))
    #         if sc > max_score:
    #             max_score, max_seq = sc, p

    #     result.append(max_seq)

    # result = torch.stack(result)

    result = [
        r.replace(bos_token, '').strip()
        for r in tokenizer.batch_decode(result[:, 2:], skip_special_tokens=True)
    ]
    # return result
    return [
        utils.get_lcs_seq(src, pred) for pred, src in zip(
            result, tokenizer.batch_decode(inp_ids, skip_special_tokens=True))
    ]
    # print(result[-1])
    # return tokenizer.batch_decode(result, skip_special_tokens=True)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--dest', default='output.csv', type=str)
    args = parser.parse_args()
    utils.set_seed(args.seed)

    test_df = pd.read_csv(args.data[0])
    if args.copy_mode:
        model = LongT5ForCopyGeneration.from_pretrained(
            args.pretrained).to('cuda')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.pretrained).to('cuda')
    model.eval()
    ds_qr = GenerationTestDataset(pd.read_csv(args.data[0]), args.pretrained,
                                  args.template_inp)
    ds_rq = GenerationTestDataset(pd.read_csv(args.data[0]), args.pretrained,
                                  ds_qr.swap_template(args.template_inp))
    tokenizer = ds_qr.tokenizer
    dl_qr = DataLoader(ds_qr, batch_size=args.batch_size)
    dl_rq = DataLoader(ds_rq, batch_size=args.batch_size)
    r_bos = "[r]"
    q_bos = "[q]"
    print(r_bos, q_bos)
    r_preds, q_preds, ids = [], [], []
    print(ds_qr.template_inp)
    for idxs, inp_ids, attn_mask in tqdm(dl_qr):
        q_preds.extend(generate(model, inp_ids, attn_mask, tokenizer, q_bos))
        ids.extend(idxs.tolist())
        
    print(ds_rq.template_inp)
    for idxs, inp_ids, attn_mask in tqdm(dl_rq):
        r_preds.extend(generate(model, inp_ids, attn_mask, tokenizer, r_bos))
    assert len(q_preds) == len(r_preds)

    result = pd.DataFrame({
        'id': ids,
        'q': [f"{q.strip()}" for q in q_preds],
        'r': [f"{r.strip()}" for r in r_preds]
    })
    result.to_csv(
        args.dest,
        sep=',',
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        #   escapechar='\\'
    )
    # with open(args.dest, 'w') as f:
    #     f.write(f'id,q\',r\'\n')
    #     for i, q, r in zip(ids, q_preds, r_preds):
    #         f.write(f'{i},"{q}","{r}"\n')
    # if ground truth exists
    if 'q\'' in test_df.columns:
        r_tgts, q_tgts, idxs = [], [], []
        # r_preds = [
        #     get_lcs_seq(tgt, pred)
        #     for tgt, pred in zip(test_df['r'].tolist(), r_preds)
        # ]
        # q_preds = [
        #     get_lcs_seq(tgt, pred)
        #     for tgt, pred in zip(test_df['q'].tolist(), q_preds)
        # ]
        print(r_preds[5], q_preds[5])
        for idx, grp in test_df.groupby('id'):
            q_tgts.append([s.replace("\"", "") for s in grp['q\''].tolist()])
            r_tgts.append([s.replace("\"", "") for s in grp['r\''].tolist()])
            idxs.append(idx)
        print("lcs_score:", utils.LCS_score(q_preds, r_preds, q_tgts, r_tgts))
        print("q_lcs_score:", utils.LCS_score_side(q_preds, q_tgts))
        print("r_lcs_score:", utils.LCS_score_side(r_preds, r_tgts))
