# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from typing import List
from functools import partial
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torchtune.modules import (
    FeedForward,
)
from torchtune.models.gemma2.gemma_rotary import Gemma2RotaryPositionalEmbeddings
from torchtune.models.gemma2.rms_norm import GemmaRMSNorm
from torchtune.models.gemma2.transformer import Gemma2TransformerDecoder
from torchtune.models.gemma2.gemma_attention import Gemma2Attention
from torchtune.models.gemma2.gemma_decoder_layer import Gemma2DecoderLayer

from torchtune.modules.peft import LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Gemma2 9B models and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``CausalSelfAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""

def gemma2(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    slidding_window_size: int, 
    attention_types: list,
    use_pre_ffw_norm: bool,
    use_post_ffw_norm: bool,
    attn_logit_softcapping: float,
    query_pre_attn_scalar: float,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    norm_embeddings: bool = True,
    ) -> Gemma2TransformerDecoder:
    """
    Build the decoder associated with the gemma model. This includes:
    - Token embeddings
    - num_layers number of TransformerDecoderLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    This does NOT currently include inference-time optimizations such as
    sliding-window attention

    Args:
       vocab_size (int): The number of tokens in the vocabulary.
       num_layers (int): The number of layers in the transformer decoder.
       num_heads (int): The number of query heads. For multi-head attention, this is also the
           number of heads for key and value.
       head_dim (int): The dimension of each head.
       num_kv_heads (int): The number of key and value heads.
       embed_dim (int): The embedding dimension for self-attention.
       intermediate_dim (int): The intermediate dimension for the MLP.
       max_seq_len (int): The maximum sequence length that the model will be run with.
       sliding_window_size (int): The size of the sliding window for attention.
       attention_types (list): A list of types of attention mechanisms to use.
       use_pre_ffw_norm (bool): Whether to use layer normalization before the feed-forward network.
       use_post_ffw_norm (bool): Whether to use layer normalization after the feed-forward network.
       attn_logit_softcapping (float): The soft capping value for attention logits.
       query_pre_attn_scalar (float): A scalar applied to the query before attention.
       norm_eps (float): Epsilon value for RMS norms. Defaults to 1e-6.
       rope_base (int): Base for the rotary positional embeddings. Defaults to 10,000.
       norm_embeddings (bool): Whether to apply layer normalization before the self-attention
           and MLP layers. Defaults to True.

    Returns:
       Gemma2TransformerDecoder: An instance of the gemma model decoder.
    """
    rope = Gemma2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    attn = Gemma2Attention(
        hidden_size=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
        attn_logit_softcapping=attn_logit_softcapping,
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        sliding_window_size=slidding_window_size,
        query_pre_attn_scalar=query_pre_attn_scalar,
    )

    mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

    if use_pre_ffw_norm :
        pre_ffw_norm = GemmaRMSNorm(embed_dim, eps=norm_eps)
    else :
        pre_ffw_norm = None

    if use_post_ffw_norm :
        post_ffw_norm = GemmaRMSNorm(embed_dim, eps=norm_eps)
    else :
        post_ffw_norm = None


    layer = Gemma2DecoderLayer(
        input_layernorm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        attn=attn,
        post_attn_layernorm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        pre_ffn_layernorm=pre_ffw_norm,
        mlp=mlp,
        post_ffn_layernorm=post_ffw_norm
    )

    
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    model = Gemma2TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        attn_types=attention_types,
        norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        norm_embeddings=norm_embeddings,
    )
    return model

def gemma_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Gemma model.

    Args:
        dim (int): input dimension to the MLP
        hidden_dim (int): hidden dimension of the MLP
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    activation = nn.GELU(approximate="tanh")
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation)


def lora_gemma2(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    *,
    # gemma2 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    sliding_window_size: int,
    attention_types: list,
    use_pre_ffw_norm: bool,
    use_post_ffw_norm: bool,
    attn_logit_softcapping: float,
    query_pre_attn_scalar: float,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    norm_embeddings: bool = True,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
) -> Gemma2TransformerDecoder:
    """
    Return a version of Gemma with LoRA applied based on the passed in configuration.

    Note that output projection LoRA is not supported because it is tied to token embeddings.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): A list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj"}``. Output projection is not supported.
        apply_lora_to_mlp (bool): Whether to apply LoRA to the MLP in each transformer layer.
            Defaults to False.
        vocab_size (int): The number of tokens in the vocabulary.
        num_layers (int): The number of layers in the transformer decoder.
        num_heads (int): The number of query heads. For multi-head attention, this is also the
            number of heads for key and value.
        head_dim (int): The dimension of each head.
        num_kv_heads (int): The number of key and value heads.
        embed_dim (int): The embedding dimension for self-attention.
        intermediate_dim (int): The intermediate dimension for the MLP.
        max_seq_len (int): The maximum sequence length that the model will be run with.
        sliding_window_size (int): The size of the sliding window for attention.
        attention_types (list): A list of types of attention mechanisms to use.
        use_pre_ffw_norm (bool): Whether to use layer normalization before the feed-forward network.
        use_post_ffw_norm (bool): Whether to use layer normalization after the feed-forward network.
        attn_logit_softcapping (float): The soft capping value for attention logits.
        query_pre_attn_scalar (float): A scalar applied to the query before attention.
        norm_eps (float): Epsilon value for RMS norms. Defaults to 1e-6.
        rope_base (int): Base for the rotary positional embeddings. Defaults to 10,000.
        norm_embeddings (bool): Whether to apply layer normalization before the self-attention
            and MLP layers. Defaults to True.
        lora_rank (int): The rank of each low-rank approximation for LoRA.
        lora_alpha (float): The scaling factor for the low-rank approximation in LoRA.
        lora_dropout (float): The dropout probability for LoRA. Defaults to 0.0.
        quantize_base (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers that LoRA is applied to. Quantization of the final output
            linear projection is not supported currently.

    Returns:
        Gemma2TransformerDecoder: An instance of the Gemma2 transformer decoder with LoRA applied to
        a subset of the attention projections in each layer.
    """
    self_attn = lora_gemma2_self_attention(
        lora_modules=lora_attn_modules,
        embed_dim=embed_dim,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_logit_softcapping=attn_logit_softcapping,
        sliding_window_size=sliding_window_size,
        query_pre_attn_scalar=query_pre_attn_scalar,
        rope_base=rope_base,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
    )

    if apply_lora_to_mlp:
        mlp = lora_gemma_mlp(
            dim=embed_dim,
            hidden_dim=intermediate_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            quantize_base=quantize_base,
            lora_dropout=lora_dropout,
        )
    else:
        mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    
    if use_pre_ffw_norm :
        pre_ffw_norm = GemmaRMSNorm(embed_dim, eps=norm_eps)
    else :
        pre_ffw_norm = None

    if use_post_ffw_norm :
        post_ffw_norm = GemmaRMSNorm(embed_dim, eps=norm_eps)
    else :
        post_ffw_norm = None


    layer = Gemma2DecoderLayer(
        input_layernorm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        attn=self_attn,
        post_attn_layernorm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        pre_ffn_layernorm=pre_ffw_norm,
        mlp=mlp,
        post_ffn_layernorm=post_ffw_norm
    )

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    model = Gemma2TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        attention_types=attention_types, 
        norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        norm_embeddings=norm_embeddings,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to higher precision, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(
                reparametrize_as_dtype_state_dict_post_hook,
                # TODO this is clowny, figure out a better way to get what precision the rest
                # of the model is in
                dtype=tok_embeddings.weight.dtype,
                offload_to_cpu=True,
            )
        )

    return model


def lora_gemma2_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # Gemma2Attention args
    embed_dim: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    max_seq_len: int,
    rope_base: int = 10_000,
    attn_logit_softcapping: float, 
    sliding_window_size: int,
    query_pre_attn_scalar: float,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
) -> Gemma2Attention:
    

    if not lora_modules:
        raise ValueError(
            f"Must pass one or more of {LORA_ATTN_MODULES} as lora_modules"
        )

    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    q_proj = (
        LoRALinear(
            embed_dim,
            num_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "q_proj" in lora_modules
        else nn.Linear(embed_dim, num_heads * head_dim, bias=False)
    )
    k_proj = (
        LoRALinear(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "k_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    v_proj = (
        LoRALinear(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "v_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    output_proj = (
        LoRALinear(
            num_heads * head_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "output_proj" in lora_modules
        else nn.Linear(num_heads * head_dim, embed_dim, bias=False)
    )

    rope = Gemma2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

    self_attn = Gemma2Attention(
        hidden_size=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        attn_logit_softcapping=attn_logit_softcapping,
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        sliding_window_size=sliding_window_size,
        query_pre_attn_scalar=query_pre_attn_scalar,
    )
    return self_attn


def lora_gemma_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
) -> FeedForward:
    gate_proj = LoRALinear(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    down_proj = LoRALinear(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    up_proj = LoRALinear(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    activation = nn.GELU(approximate="tanh")

    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation)
