from typing import Dict, List
import os, json
import pandas as pd
import torch
from transformers import (DefaultDataCollator, EvalPrediction,
                          HfArgumentParser, AutoModelForSeq2SeqLM, T5Tokenizer,
                          Trainer, TrainingArguments)
from custom_trainner import GenerationTrainer, GenerationForceTrainer
from dataset import GenerationDataset, split_train_test_df, CopyDataset
from opt import get_parser
from utils import LCS_score, set_seed
from model import LongT5ForCopyGeneration
from torch.utils import data

class TaskTrainingArguments(TrainingArguments):
    def set_args(self, args):
        self.evaluation_strategy = 'epoch'
        self.do_eval: bool = True
        self.per_device_train_batch_size: int = args.batch_size
        self.per_device_eval_batch_size: int = args.batch_size
        self.gradient_accumulation_steps: int = args.gradient_accumulation_steps
        self.learning_rate = args.lr
        self.warmup_steps = 4000
        self.save_strategy = 'epoch'
        # self.label_smoothing_factor = 0.1
        self.num_train_epochs = 10
        self.logging_first_step = True
        self.ignore_data_skip=True
        self.weight_decay = 0.01
        self.gradient_checkpointing = False


def get_datacolator(pad):
    print('pad_token', pad)

    def t2tdata_collator(batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([b['input_ids'] for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])
        tr_pos = len(input_ids[0]) - 1
        while (input_ids[:, tr_pos] != 0).sum() == 0:
            tr_pos -= 1
        tr_pos += 1
        input_ids, attention_mask = input_ids[:, :
                                              tr_pos], attention_mask[:, :
                                                                      tr_pos]
        if 'decoder_input_ids' in batch[0]:
            dec_inp_ids = torch.stack([b['decoder_input_ids'] for b in batch])
            # decoder_attention_mask = torch.stack(
            #     [b['decoder_attention_mask'] for b in batch])
        else:
            dec_inp_ids, decoder_attention_mask = None, None
        lm_labels = torch.stack([b['labels'] for b in batch])
        lm_labels[lm_labels == pad] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': lm_labels,
            # 'decoder_attention_mask': decoder_attention_mask,
            'decoder_input_ids': dec_inp_ids
        }

    return t2tdata_collator


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    val_df = pd.read_csv(args.val_data)
    # train_df, val_df = split_train_test_df(df, 0.1)

    if not args.copy_mode:
        model_calss = AutoModelForSeq2SeqLM
        dataset_class = GenerationDataset
    else:
        model_calss = LongT5ForCopyGeneration
        dataset_class = CopyDataset
    print(model_calss, dataset_class)
    model = model_calss.from_pretrained(args.pretrained)
    train_datasets = [dataset_class(pd.read_csv(path),
                                  args.pretrained,
                                  template_inp=args.template_inp,
                                  aug_prob=args.aug_prob) for path in args.data]
    valid_dataset = dataset_class(val_df,
                                  args.pretrained,
                                  template_inp=args.template_inp,
                                  aug_prob=0)
    train_dataset = data.ConcatDataset(train_datasets)
    model.resize_token_embeddings(len(train_datasets[0].tokenizer))
    train_args = TaskTrainingArguments(args.output_dir)
    train_args.set_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    trainer = GenerationTrainer(
        args=train_args,
        model=model.cuda(),
        train_dataset=train_dataset,
        tokenizer=train_datasets[0].tokenizer,
        eval_dataset=valid_dataset,
        data_collator=get_datacolator(train_datasets[0].tokenizer.pad_token_id),
    )
    trainer.train()