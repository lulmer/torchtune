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
    CausalSelfAttention,
    FeedForward,
    RotaryPositionalEmbeddings,
    TransformerDecoderLayer,
)
from torchtune.models.gemma2.rms_norm import GemmaRMSNorm
from torchtune.models.gemma2.transformer import Gemma2TransformerDecoder
from torchtune.models.gemma2.gemma_attention import Gemma2Attention
from torchtune.models.gemma2.gemma_decoder_layer import Gemma2DecoderLayer

from torchtune.modules.peft import LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Gemma 2B models and popular variants such as LoRA.

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
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads.
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms Default: 1e-6
        rope_base (int): base for the rotary positional embeddings. Default: 10_000
        norm_embeddings (bool): whether to apply layer norm before the self-attention
            and mlp layers. Default: True

    Returns:
        GemmaTransformerDecoder: Instantiation of gemma model.
    """
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_att = Gemma2Attention(
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

    # self.qkv_proj = nn.Linear(hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim)
    # self.o_proj = nn.Linear(
    #         self.num_heads * self.head_dim,
    #         self.hidden_size,
    #         quant=quant)
    ###########################################
    # qkv = self.qkv_proj(hidden_states)
    # xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],
    #                         dim=-1)

    # xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
    # xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    # xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

    mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

    if use_pre_ffw_norm :
        pre_ffw_norm = GemmaRMSNorm(embed_dim, eps=norm_eps)
    else :
        pre_ffw_norm = None

    if use_post_ffw_norm :
        pre_post_norm = GemmaRMSNorm(embed_dim, eps=norm_eps)
    else :
        pre_post_norm = None


    layer = Gemma2DecoderLayer(
        input_layernorm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        attn=self_att,
        post_attn_layernorm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        pre_ffn_layernorm=pre_ffw_norm,
        mlp=mlp,
        post_ffn_layernorm=pre_post_norm
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


def lora_gemma(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    *,
    # gemma args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
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
    Note: output projection lora is not supported because it is tied to token embeddings

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads.
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms Default: 1e-6
        rope_base (int): base for the rotary positional embeddings. Default: 10_000
        norm_embeddings (bool): whether to apply layer norm before the self-attention
            and mlp layers. Default: True
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma model with LoRA applied to
        a subset of the attention projections in each layer.
    """
    self_attn = lora_gemma_self_attention(
        lora_modules=lora_attn_modules,
        embed_dim=embed_dim,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
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

    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    model = Gemma2TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
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


def lora_gemma_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # CausalSelfAttention args
    embed_dim: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 10_000,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
) -> CausalSelfAttention:
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

    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=rope,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
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
