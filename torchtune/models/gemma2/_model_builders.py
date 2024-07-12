# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.models.gemma2._component_builders import gemma2, lora_gemma2
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
        vocab_size=256_000,
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

def lora_gemma2_9b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> Gemma2TransformerDecoder:
    """
    Builder for creating a Gemma 2B model with LoRA enabled.

    The Gemma defaults are the same as in :func:`~torchtune.models.gemma.gemma_2b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 2B model with LoRA applied
    """
    return lora_gemma2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=256_000,
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
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )

qlora_gemma2_9b = partial(lora_gemma2_9b, quantize_base=True)

qlora_gemma2_9b.__doc__ = """
Builder for creating a Gemma2 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma2_9b` for full API arguments.
"""


def gemma2_27b() -> Gemma2TransformerDecoder:
    """
    Builder for creating a Gemma 2B model initialized w/ the default 2b parameter values
    from: https://blog.google/technology/developers/gemma-open-models/

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 2B model
    """
    return gemma2(
        vocab_size=256_000,
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

def lora_gemma2_27b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> Gemma2TransformerDecoder:
    """
    Builder for creating a Gemma 2B model with LoRA enabled.

    The Gemma defaults are the same as in :func:`~torchtune.models.gemma.gemma_2b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 2B model with LoRA applied
    """
    return lora_gemma2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=256_000,
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
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )

qlora_gemma2_27b = partial(lora_gemma2_27b, quantize_base=True)

qlora_gemma2_27b.__doc__ = """
Builder for creating a Gemma2 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma2_27b` for full API arguments.
"""

def gemma_tokenizer(path: str) -> SentencePieceTokenizer:
    """
    Tokenizer for Gemma2.

    Args:
        path (str): path to the tokenizer

    Returns:
        SentencePieceTokenizer: Instantiation of the Gemma tokenizer
    """
    tokenizer = SentencePieceTokenizer(path)
    tokenizer.pad_id = 0
    return tokenizer





