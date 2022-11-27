from transformers import T5ForConditionalGeneration, T5Config
from torch import nn
# n_positions

class T5ForCopyGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.d_model, config.n_positions, bias=False)
        self.post_init()