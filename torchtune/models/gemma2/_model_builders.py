# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.models.gemma2._component_builders import gemma2, lora_gemma
from torchtune.models.gemma2.transformer import Gemma2TransformerDecoder

from torchtune.modules.tokenizers import SentencePieceTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES

from functools import partial

"""
Model builders build specific instantiations using component builders. For example
the ``gemma_2b`` model builder uses the ``gemma`` component builder.
"""


def gemma2_9b() -> Gemma2TransformerDecoder:
    """
    Builder for creating a Gemma 2B model initialized w/ the default 2b parameter values
    from: https://blog.google/technology/developers/gemma-open-models/

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 2B model
    """
    return gemma2(
        vocab_size=256_128,
        num_layers=42,
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        embed_dim=3584, #also refered as hidden size
        intermediate_dim=14336,
        max_seq_len=8192,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        #final_logit_softcapping=30.0, # To be added somewhere in the decoding process
        attn_logit_softcapping=50.0,
        slidding_window_size=4096,
        attention_types=['LOCAL_SLIDING', 'GLOBAL'] * 21,
        norm_eps=1e-6,
        query_pre_attn_scalar=224,  # hidden_size / num_attention_heads
    )

def gemma2_27b() -> Gemma2TransformerDecoder:
    """
    Builder for creating a Gemma 2B model initialized w/ the default 2b parameter values
    from: https://blog.google/technology/developers/gemma-open-models/

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 2B model
    """
    return gemma2(
        vocab_size=256_128,
        num_layers=46,
        num_heads=32,
        num_kv_heads=16,
        head_dim=128,
        embed_dim=4608, #also refered as hidden size
        intermediate_dim=36864,
        max_seq_len=8192,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        #final_logit_softcapping=30.0, # To be added somewhere in the decoding process
        attn_logit_softcapping=50.0,
        slidding_window_size=4096,
        attention_types=['LOCAL_SLIDING', 'GLOBAL'] * 23,
        norm_eps=1e-6,
        query_pre_attn_scalar=144,  # hidden_size / num_attention_heads
    )

def gemma_tokenizer(path: str) -> SentencePieceTokenizer:
    """
    Tokenizer for Gemma.

    Args:
        path (str): path to the tokenizer

    Returns:
        SentencePieceTokenizer: Instantiation of the Gemma tokenizer
    """
    tokenizer = SentencePieceTokenizer(path)
    tokenizer.pad_id = 0
    return tokenizer





