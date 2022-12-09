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
from typing import List, Tuple


def find_best_result(beam_result: List[Tuple[str, str]], original_text: str):
    # scores = [
    #     utils.LCS_score_side([r], [[original_text]]) for r in beam_result
    # ]
    # print(original_text)
    # print(beam_result)
    # print(scores)
    # idx = scores.index(max(scores))
    
    beam_result = [(utils.get_lcs_seq(original_text, r), s) for r, s in beam_result]
    # print(beam_result)
    # lens = list(map(lambda x: len(x.split()), beam_result))
    scores = [
         s for r, s in beam_result
    ]

    idx = scores.index(max(scores))
    # print(beam_result[idx])
    return beam_result[idx][0]
    # return utils.get_lcs_seq(original_text, beam_result[idx])


def generate(model: AutoModelForSeq2SeqLM,
             inp_ids: torch.Tensor,
             attn_mask: torch.Tensor,
             tokenizer,
             bos_token,
             text,
             device='cuda',
             num_seqs=5,
             **kwargs):
    assert inp_ids.shape[0] == len(text)

    tr_pos = len(inp_ids[0]) - 1
    while (inp_ids[:, tr_pos] != 0).sum() == 0:
        tr_pos -= 1
    tr_pos += 1
    inp_ids, attn_mask = inp_ids[:, :tr_pos], attn_mask[:, :tr_pos]

    r_bos = tokenizer.encode('[r]', add_special_tokens=False)[0]
    q_bos = tokenizer.encode('[q]', add_special_tokens=False)[0]
    vocab = inp_ids.clone().detach()
    # q_idx = (vocab == q_bos).nonzero()[1::2]
    # r_idx = (vocab == r_bos).nonzero()[1::2]

    if bos_token == "[r]":
        bos_token_id = r_bos
        # for r in r_idx:
        #     vocab[r[0], :r[1]] = tokenizer.pad_token_id
    else:  # [q]
        # [q] .. [r]
        bos_token_id = q_bos
        # for r in q_idx:
        #     vocab[r[0], :r[1]] = tokenizer.pad_token_id

    # bos_token_id = r_bos if bos_token == '[r]' else q_bos

    inp_ids, attn_mask = inp_ids.to(device), attn_mask.to(device)
    model.config.output_scores = True
    beam_result = model.generate(
        inp_ids,
        attention_mask=attn_mask,
        max_new_tokens=150,
        # early_stopping=True,
        logits_processor=LogitsProcessorList(
            [utils.RestrictWordsProcessor(vocab, len(tokenizer))]),
        renormalize_logits=True,
        # forced_bos_token_id=bos_token_id,
        # decoder_start_token_id=bos_token_id,
        forced_decoder_ids=[[1, bos_token_id]],
        num_beams=num_seqs,
        # repetition_penalty=1.3,
        # num_beam_groups=5,
        num_return_sequences=num_seqs,
        do_sample=True,
        top_p=0.15,
        temperature=1.2,
        return_dict_in_generate=True, output_scores=True,
        **kwargs)
    # print(beam_result)
    scores, result = beam_result.sequences_scores, beam_result.sequences
    # scores = torch.softmax(scores, -1)
    scores = scores.exp()
    # print(scores)
    
    result = tokenizer.batch_decode(result[:, 2:],
                                    skip_special_tokens=True)
    result = [r.replace(bos_token, '').strip() for r in result]
    best_result = []
    batch = []
    for r, s in zip(result, scores):
        batch.append((r, s))
        if len(batch) == num_seqs:
            best_result.append(find_best_result(batch, text[len(best_result)]))
            batch = []
    if len(batch) != 0:
        best_result.append(find_best_result(batch, text[len(best_result)]))

    return best_result


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
        q_text = [ds_qr.get_qr(i)[0] for i in idxs]
        q_preds.extend(
            generate(model, inp_ids, attn_mask, tokenizer, q_bos, q_text))
        ids.extend(idxs.tolist())

    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained).to('cuda')

    model.eval()
    print(ds_rq.template_inp)
    for idxs, inp_ids, attn_mask in tqdm(dl_rq):
        r_text = [ds_rq.get_qr(i)[1] for i in idxs]
        r_preds.extend(
            generate(model, inp_ids, attn_mask, tokenizer, r_bos, r_text))

    print(len(q_preds), len(r_preds), len(ids))
    assert len(q_preds) == len(r_preds)
    assert len(ids) == len(r_preds)

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
        print(r_preds[5], q_preds[5])
        for idx, grp in test_df.groupby('id'):
            q_tgts.append([s.replace("\"", "") for s in grp['q\''].tolist()])
            r_tgts.append([s.replace("\"", "") for s in grp['r\''].tolist()])
            idxs.append(idx)
        print("lcs_score:", utils.LCS_score(q_preds, r_preds, q_tgts, r_tgts))
        print("q_lcs_score:", utils.LCS_score_side(q_preds, q_tgts))
        print("r_lcs_score:", utils.LCS_score_side(r_preds, r_tgts))
