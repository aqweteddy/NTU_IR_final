from transformers import Trainer
import torch.nn.functional as F
import utils, torch


class GenerationTrainer(Trainer):
    def compute_loss(self, model, inputs: dict, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs, use_cache=False)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1),
                               label_smoothing=0.15,
                               ignore_index=-100)
        return (loss, outputs) if return_outputs else loss


class GenerationForceTrainer(Trainer):
    def compute_loss(self, model, inputs: dict, return_outputs=False):
        labels = inputs.get("labels")  # [B, S]
        input_ids = inputs.get('input_ids')
        outputs = model(**inputs)
        logits = outputs.get("logits") # [B, S, V]
        with torch.no_grad():
            restricted_vocab = F.one_hot(
            input_ids.detach(),
            num_classes=model.config.vocab_size).to(input_ids.device).sum(1).bool()  # B, V
            labels = F.one_hot(labels, num_classes=model.config.vocab_size).to(input_ids.device).float() # [B, S, V]
            mask = restricted_vocab # B, V
            mask = mask.unsqueeze(1).repeat(1, labels.shape[1], 1)
            print(mask.shape)
            labels[mask] = 0.2
        loss = F.cross_entropy(logits,
                               labels,
                               label_smoothing=0.15,
                               ignore_index=-100)
        print(loss)
        return (loss, outputs) if return_outputs else loss
