from transformers import Trainer
import torch.nn.functional as F
import utils


class GenerationTrainer(Trainer):
    def compute_loss(self, model, inputs: dict, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1),
                               label_smoothing=0.2)
        return (loss, outputs) if return_outputs else loss