from typing import Dict, List
import os, json
import pandas as pd
import torch
from transformers import (DefaultDataCollator, EvalPrediction, HfArgumentParser,
                          T5ForConditionalGeneration, T5Tokenizer, Trainer,
                          TrainingArguments)

from dataset import Dataset, split_train_test_df
from opt import get_parser
from utils import LCS_score, set_seed
from dataclasses import dataclass

class TaskTrainingArguments(TrainingArguments):
    def set_args(self, args):
        self.evaluation_strategy = 'epoch'
        self.do_eval: bool = True
        self.per_device_train_batch_size: int = args.batch_size
        self.per_device_eval_batch_size : int = args.batch_size
        self.gradient_accumulation_steps  : int = args.gradient_accumulation_steps 
        self.learning_rate = args.lr
        self.warmup_ratio = 0.1
        self.save_strategy = 'epoch'
        # self.label_smoothing_factor = 0.1
        self.num_train_epochs = 5
        self.logging_first_step = True

def t2tdata_collator(batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    # inp_ids, inp_mask, tgt_ids, tgt_mask = batch
    input_ids = torch.stack([b[0] for b in batch])
    lm_labels = torch.stack([b[2] for b in batch])
    # dec_inp_ids = torch.stack([b[2] for b in batch])
    lm_labels[lm_labels[:, :] == 0] = -100
    attention_mask = torch.stack([b[1] for b in batch])
    # decoder_attention_mask = torch.stack([b[3] for b in batch])
    # print(lm_labels)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        # 'decoder_attention_mask': decoder_attention_mask,
        # 'decoder_input_ids': dec_inp_ids
    }



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    model = T5ForConditionalGeneration.from_pretrained(
        args.pretrained
    )
    df = pd.read_csv(args.data)
    train_df, test_df = split_train_test_df(df, 0.1)
    train_dataset = Dataset(train_df, args.pretrained, template_inp=args.template_inp)
    valid_dataset = Dataset(test_df, args.pretrained, template_inp=args.template_inp)
    valid_dataset.tokenizer = train_dataset.tokenizer
    model.resize_token_embeddings(len(train_dataset.tokenizer))
    train_args = TaskTrainingArguments(args.output_dir)
    train_args.set_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
         json.dump(vars(args), f, ensure_ascii=False, indent=2)
    trainer = Trainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        tokenizer=train_dataset.tokenizer,
        eval_dataset=valid_dataset,
        data_collator=t2tdata_collator,
        # compute_metrics=,
    )
    trainer.train()
    trainer.save_model()

