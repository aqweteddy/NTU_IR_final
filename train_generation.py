from typing import Dict, List
import os, json
import pandas as pd
import torch
from transformers import (DefaultDataCollator, EvalPrediction,
                          HfArgumentParser, AutoModelForSeq2SeqLM,
                          T5Tokenizer, Trainer, TrainingArguments)
from custom_trainner import GenerationTrainer
from dataset import GenerationDataset, split_train_test_df
from opt import get_parser
from utils import LCS_score, set_seed


class TaskTrainingArguments(TrainingArguments):
    def set_args(self, args):
        self.evaluation_strategy = 'epoch'
        self.do_eval: bool = True
        self.per_device_train_batch_size: int = args.batch_size
        self.per_device_eval_batch_size: int = args.batch_size
        self.gradient_accumulation_steps: int = args.gradient_accumulation_steps
        self.learning_rate = args.lr
        self.warmup_ratio = 0.1
        self.save_strategy = 'epoch'
        # self.label_smoothing_factor = 0.1
        self.num_train_epochs = 5
        self.logging_first_step = True
        self.weight_decay=0.01

def get_datacolator(pad):
    print('pad_token', pad)
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
        lm_labels[lm_labels == pad] = -100
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
    return t2tdata_collator


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained)
    df = pd.read_csv(args.data)
    train_df, test_df = split_train_test_df(df, 0.1)
    train_dataset = GenerationDataset(train_df,
                                      args.pretrained,
                                      template_inp=args.template_inp,
                                      aug_prob=args.aug_prob)
    valid_dataset = GenerationDataset(test_df,
                                      args.pretrained,
                                      template_inp=args.template_inp,
                                      aug_prob=args.aug_prob)
    valid_dataset.tokenizer = train_dataset.tokenizer
    model.resize_token_embeddings(len(train_dataset.tokenizer))
    train_args = TaskTrainingArguments(args.output_dir)
    train_args.set_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    trainer = GenerationTrainer(
        args=train_args,
        model=model,
        train_dataset=train_dataset,
        tokenizer=train_dataset.tokenizer,
        eval_dataset=valid_dataset,
        data_collator=get_datacolator(train_dataset.tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model()
