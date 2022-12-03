from typing import Optional, List, Tuple, Union
from transformers import LongT5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
import torch
from torch import nn
# n_positions


class LongT5ForCopyGeneration(LongT5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

        self.copy_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(32102, 1024, bias=False))
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.copy_head(self.lm_head(sequence_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)),
                            labels.reshape(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits, ) + decoder_outputs[1:] + encoder_outputs
            return ((loss, ) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def batch_greedy(
        self,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens=20,
        forced_decoder_ids=None,
        pad_token_id=None,
        eos_token_id=None
    ):
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        result = [[pad_token_id]] * len(inputs)  # [BOS]
        eos_fl = [False] * len(inputs)
        model_kwargs = {}
        model_kwargs['encoder_outputs'] = self.encoder(
            input_ids=inputs, attention_mask=attention_mask)

        while not all(eos_fl) and len(result[0]) <= max_new_tokens:
            inps = torch.LongTensor(result).to(inputs.device)
            next_index_logits = self(
                inps,
                **model_kwargs,
                return_dict=True
            )['logits'][:, -1]
            
            next_tokens = next_logits.argmax(-1)
            for i in range(len(result)):
                if eos_fl[i] == True:
                    result[i].append(0)
                    # print(result[i])
                else:
                    result[i].append(next_tokens[i].item())
                    if result[i][-1] == eos_token_id:  # EOS
                        eos_fl[i] = True
        return result

    @torch.no_grad()
    def generate(self,
                 inputs: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens=20,
                 forced_decoder_ids=None,
                 **args) -> torch.LongTensor:
        model_kwargs = {}
        model_kwargs['encoder_outputs'] = self.encoder(
            input_ids=inputs, attention_mask=attention_mask)
        input_ids = torch.zeros((inputs.shape[0], 1),
                                dtype=torch.long).to(inputs.device)
        return self.greedy_search(
            inputs,
            input_ids,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=max_new_tokens,
            model_kwargs=model_kwargs,
        )

    def greedy_search(self,
                      input: torch.LongTensor,
                      input_ids: torch.LongTensor,
                      max_new_tokens=None,
                      forced_decoder_ids=None,
                      pad_token_id: Optional[int] = None,
                      eos_token_id: Optional[int] = None,
                      output_attentions: Optional[bool] = None,
                      output_hidden_states: Optional[bool] = None,
                      model_kwargs=None,
                      **kwargs) -> torch.LongTensor:
        # return super().greedy_search(input_ids, logits_processor, stopping_criteria, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus, **model_kwargs)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        now_pos_ids = [0] * input.shape[0]
        batch_ids = torch.arange(input.shape[0], device=input.device)

        while True:
            for pos, id in forced_decoder_ids:
                if len(input_ids[1]) == pos:
                    fill = torch.zeros((input_ids.shape[0], 1)).to(
                        input_ids.device).float() + id
                    input_ids = torch.cat([input_ids, fill.long()], dim=-1)
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            next_index_logits = outputs.logits[:, -1, :]
            for i, p in enumerate(now_pos_ids):
                next_index_logits[i, 0:p + 1] = float('-inf')

            next_index = torch.argmax(next_index_logits,
                                      dim=-1).detach().cpu().tolist()  # B
            next_index = [
                idx if idx < len(input[0]) else pad_token_id
                for idx in next_index
            ]
            now_pos_ids = [
                next_index[i]
                if next_index[i] != pad_token_id else now_pos_ids[i]
                for i in range(len(next_index))
            ]
            next_tokens = input[batch_ids, next_index]
            next_tokens = next_tokens.to(unfinished_sequences.device)
            # next_tokens = next_tokens * unfinished_sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != eos_token_id).long())
            if unfinished_sequences.max(
            ) == 0 or input_ids.shape[1] >= max_new_tokens:
                break

        return input_ids


if __name__ == '__main__':
    # import pandas as pd
    # from dataset import CopyDataset
    # df = pd.read_csv('data/train.csv')
    # ds = CopyDataset(df, 't5-base', aug_prob=0.)
    input_ids = torch.randint(0, 1000, size=(5, 512))
    labels = torch.stack([torch.arange(5)] * 5)
    decoder_input_ids = torch.randint(0, 1000, size=(5, 7))
    print(input_ids.shape, labels.shape)
    print(input_ids)
    model = LongT5ForCopyGeneration.from_pretrained(
        '../ckpt/long_t5_base_copy/checkpoint-29312')
    # model = LongT5ForCopyGeneration.from_pretrained('../ckpt/long_t5_base_copy/checkpoint-29368')

    # print(model(input_ids, labels=labels).loss)
    print(model.generate(input_ids, forced_decoder_ids=[[1, 200]]))
