# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Union, List
import logging

import torch
from torch import nn
import torch.utils.checkpoint

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEmbeddings,
    BertPooler,
    BertConfig,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    BertLayer,
    BertPreTrainingHeads,
    # BertPretrainingHeads,
)

from ..misc.file_utils import cached_path
from .constants import (
    PRETRAINED_CONFIG_ARCHIVE_MAP,
    PRETRAINED_MODEL_ARCHIVE_MAP,
    BERT_CONFIG_NAME,
)

# from transformers import BertPretrainingHeads

logger = logging.getLogger(__name__)


class ZenConfig(BertConfig):
    def __init__(self, num_word_hidden_layers=0, ngram_vocab_size=21128, **kwargs):
        super().__init__(**kwargs)
        self.num_word_hidden_layers = num_word_hidden_layers
        self.ngram_vocab_size = ngram_vocab_size


class ZenEncoder(nn.Module):
    """The Bert model taken from the HuggingFace Transformers library.

    The original source code can be found at transformers/models/bert/modeling_bert.py.

    I have made some minor modifications to the original code:
        - Removed the gradient checkpointing feature as it is not used.
    """

    def __init__(self, config):
        super().__init__()
        if config.num_hidden_layers < config.num_word_hidden_layers:
            raise ValueError(
                f"config.num_hidden_layers={config.num_hidden_layers} should be greater than or equal to "
                f"config.num_word_hidden_layers={config.num_word_hidden_layers}"
            )

        self.config = config
        self.main_layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.word_layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_word_hidden_layers)]
            + [None] * (config.num_hidden_layers - config.num_word_hidden_layers)
        )
        # self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        ngram_hidden_states: torch.Tensor,  # ngram-specific
        attention_mask: Optional[torch.FloatTensor] = None,
        ngram_attention_mask: Optional[torch.FloatTensor] = None,  # ngram-specific
        ngram_position_matrix: Optional[torch.LongTensor] = None,  # ngram-specific
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        next_decoder_cache = () if use_cache else None
        for i, (main_layer_module, word_layer_module) in enumerate(
            zip(self.main_layer, self.word_layer)
        ):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if word_layer_module is not None:
                ngram_layer_outputs = word_layer_module(
                    ngram_hidden_states,
                    ngram_attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                    # TODO: figure out whether this is needed in ZEN
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                )
                ngram_hidden_states = ngram_layer_outputs[0]

            layer_outputs = main_layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            _dtype = hidden_states.dtype
            hidden_states = (
                # main_hidden_states
                layer_outputs[0]
                # ngram_hidden_states
                + torch.bmm(
                    ngram_position_matrix.to(dtype=_dtype),
                    ngram_hidden_states.to(dtype=_dtype),
                )
            )
            if use_cache:
                # Warning: I have not tested this code path, so it may not work as expected
                next_decoder_cache += (layer_outputs[-1], ngram_layer_outputs[-1])
            if output_attentions:
                # Warning: I have not tested this code path, so it may not work as expected
                all_self_attentions = all_self_attentions + (
                    layer_outputs[1],
                    ngram_layer_outputs[-1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[2],
                        ngram_layer_outputs[2],
                    )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ZenNgramEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.

    Almost the same as BertEmbeddings, but is different in the following aspects:
        - No position embeddings
        - the vocab size is the ngram vocab size
    """

    def __init__(self, config: ZenConfig):
        super().__init__()
        self.ngram_vocab_size = config.ngram_vocab_size
        self.word_embeddings = nn.Embedding(
            config.ngram_vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        # if self.position_embedding_type == "absolute":
        #     position_embeddings = self.position_embeddings(position_ids)
        #     embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ZenPreTrainedModel(BertPreTrainedModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir=None):
        assert pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP

        archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]

        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file
                    )
                )
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                        archive_file,
                    )
                )
                return None

        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info(
                "loading weights file {} from cache at {}".format(
                    archive_file, resolved_archive_file
                )
            )

        model = cls(config)
        # if state_dict is None:
        state_dict = torch.load(resolved_archive_file, map_location="cpu")
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(
            s.startswith("bert.") for s in state_dict.keys()
        ):
            start_prefix = "bert."
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return model


class ZenModel(ZenPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _no_split_modules = ["BertEmbeddings", "ZenNgramEmbeddings", "BertLayer"]

    def __init__(self, config: ZenConfig, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        # # TODO: learn what is the `position_embedding_type`
        self.word_embeddings = ZenNgramEmbeddings(config)
        self.encoder: ZenEncoder = ZenEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_ngram_ids: Optional[torch.Tensor] = None,  # ngram-specific
        attention_mask: Optional[torch.Tensor] = None,
        ngram_attention_mask: Optional[torch.Tensor] = None,  # ngram-specific
        token_type_ids: Optional[torch.Tensor] = None,
        ngram_token_type_ids: Optional[torch.Tensor] = None,  # ngram-specific
        position_ids: Optional[torch.Tensor] = None,
        ngram_position_matrix: Optional[torch.Tensor] = None,  # ngram-specific
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, target_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            ngran_input_shape = input_ngram_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            ngran_input_shape = input_ngram_ids.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            # position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        ngram_embedding_output = self.word_embeddings(
            input_ids=input_ngram_ids,
            # we are not using absolute position embeddings for ngrams,
            position_ids=None,
            token_type_ids=ngram_token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )
        if ngram_attention_mask is None:
            ngram_attention_mask = torch.ones_like(input_ngram_ids, device=device)

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape
            )
        extended_ngram_attention_mask = self.get_extended_attention_mask(
            ngram_attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            ngram_embedding_output,  # ngram-specific
            attention_mask=extended_attention_mask,
            ngram_attention_mask=extended_ngram_attention_mask,  # ngram-specific
            ngram_position_matrix=ngram_position_matrix,  # ngram-specific
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ZenForPreTraining(ZenPreTrainedModel):
    """ZEN model with pre-training heads.
    This module comprises the ZEN model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        `input_ngram_ids`: input_ids of ngrams.
        `ngram_token_type_ids`: token_type_ids of ngrams.
        `ngram_attention_mask`: attention_mask of ngrams.
        `ngram_position_matrix`: position matrix of ngrams.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    """

    def __init__(self, config, output_attentions=False, keep_multihead_output=False):
        super(ZenForPreTraining, self).__init__(config)
        self.output_attentions = output_attentions
        # self.bert = ZenModel(config, output_attentions=output_attentions,
        #                       keep_multihead_output=keep_multihead_output)
        self.bert: ZenModel = ZenModel(config)

        # self.cls = ZenPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.cls = BertPreTrainingHeads(config)
        # self.apply(self.init_bert_weights)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def save_pretrained(self, save_directory, cur_index=0):
        """Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `from_pretrained()` class method.
        """
        import os

        assert os.path.isdir(save_directory), (
            "Saving path should be a directory where the model and configuration can be saved"
        )
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(
            save_directory, f"pytorch_model_{cur_index}.bin"
        )
        output_config_file = os.path.join(save_directory, f"config_{cur_index}.json")

        torch.save(self.state_dict(), output_model_file)
        self.config.to_json_file(output_config_file)
        logger.info("Model weights saved in {}".format(output_model_file))
        logger.info("Configuration saved in {}".format(output_config_file))

    def forward(
        self,
        input_ids,
        input_ngram_ids,
        ngram_position_matrix,
        token_type_ids=None,
        ngram_token_type_ids=None,
        attention_mask=None,
        ngram_attention_mask=None,
        masked_lm_labels=None,
        next_sentence_label=None,
        head_mask=None,
    ):
        dtype = next(self.parameters()).dtype

        outputs = self.bert(
            input_ids,
            input_ngram_ids,
            attention_mask=attention_mask,
            ngram_attention_mask=ngram_attention_mask,
            ngram_position_matrix=ngram_position_matrix,
            token_type_ids=token_type_ids,
            ngram_token_type_ids=ngram_token_type_ids,
            # ngram_attention_mask,
            output_attentions=False,
        )
        # if self.output_attentions:
        #     all_attentions, sequence_output, pooled_output = outputs
        # else:
        # sequence_output, pooled_output = outputs
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        if self.output_attentions:
            all_attentions = outputs.attentions

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )
        # print("[debug] prediction_scores.shape:", prediction_scores.shape)
        total_loss = 0
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        if masked_lm_labels is not None:
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            )
            total_loss += masked_lm_loss

        if next_sentence_label is not None:
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            )
            total_loss += next_sentence_loss

        if not (masked_lm_labels is None and next_sentence_label is None):
            return total_loss

        if self.output_attentions:
            return all_attentions, prediction_scores, seq_relationship_score
        return prediction_scores, seq_relationship_score


class ZenForSequenceClassification(BertPreTrainedModel):
    """ZEN model for classification.
    This module is composed of the ZEN model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model
        `output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
        `keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
            This can be used to compute head importance metrics. Default: False
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
        `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
            It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        `input_ngram_ids`: input_ids of ngrams.
        `ngram_token_type_ids`: token_type_ids of ngrams.
        `ngram_attention_mask`: attention_mask of ngrams.
        `ngram_position_matrix`: position matrix of ngrams.

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    """

    def __init__(
        self, config, num_labels=2, output_attentions=False, keep_multihead_output=False
    ):
        super(ZenForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = ZenModel(
            config,
            output_attentions=output_attentions,
            keep_multihead_output=keep_multihead_output,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        input_ngram_ids,
        ngram_position_matrix,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        outputs = self.bert(
            input_ids,
            input_ngram_ids,
            ngram_position_matrix,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False,
            head_mask=head_mask,
        )
        if self.output_attentions:
            all_attentions, _, pooled_output = outputs
        else:
            _, pooled_output = outputs
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits
