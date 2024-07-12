import torch 
from typing import Optional
from torchtune.modules.kv_cache import KVCache
from torchtune.modules import (
    RotaryPositionalEmbeddings,
)
import torch.nn as nn
import torch.nn.functional as F
import math

class Gemma2AttentionSdpa(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 q_proj: torch.Tensor,
                 k_proj: torch.Tensor,
                 v_proj: torch.Tensor,
                 output_proj: torch.Tensor,
                 max_seq_len: int,
                 pos_embeddings: RotaryPositionalEmbeddings,
                 attn_logit_softcapping: Optional[float],
                 query_pre_attn_scalar: Optional[int],
                 kv_cache: Optional[KVCache] = None,
                 sliding_window_size: Optional[int] = None,
                 ) -> None:
        super().__init__()
        # State dimensions and head dimensions
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        assert self.num_heads % self.num_kv_heads == 0, 'The number of heads must be divisible by the number of key-value heads.'
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Projections for query, key, value and output and KV cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.kv_cache = kv_cache

        # Attention enhancements
        self.attn_logit_softcapping = attn_logit_softcapping
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.pos_embeddings = pos_embeddings
        self.sliding_window_size = sliding_window_size

        # Attention scalling
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5
    
    def forward(self, 
                x: torch.Tensor,
                *,
                mask: torch.Tensor,
                attn_type: str = 'LOCAL_SLIDING',
                input_pos: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        hidden_states_shape = x.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        if input_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({input_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # Resizing the tensors prior to ROPE
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = self.pos_embeddings(xq, input_pos=input_pos)
        xk = self.pos_embeddings(xk, input_pos=input_pos)

        # Update key-value cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, xk, xv)

        # shape: [b, 1, s, s]
        if mask is not None:
            mask = mask[:, None, :, :]


        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv,
                                            self.num_queries_per_kv,
                                            dim=2)
        
        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        output = scaled_dot_product_attention_w_logits_softcapping(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=self.kv_cache is None and mask is None,
            scale=self.scaling,
            attn_logit_softcapping=self.attn_logit_softcapping,
            sliding_window_attn=attn_type == 'LOCAL_SLIDING',
            sliding_window_size=self.sliding_window_size)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.output_proj(output)
        return output


def scaled_dot_product_attention_w_logits_softcapping(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, attn_logit_softcapping=None, sliding_window_attn=False, sliding_window_size=0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        if sliding_window_attn :
            temp_mask = torch.triu(
                temp_mask, -1 * sliding_window_size + 1
            )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if sliding_window_attn :
            attn_mask = torch.triu(
                attn_mask, -1 * sliding_window_size + 1
            ) * torch.tril(attn_mask, sliding_window_size - 1)

        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if attn_logit_softcapping is not None:
            attn_weight = attn_weight / attn_logit_softcapping
            attn_weight = torch.tanh(attn_weight)
            attn_weight = attn_weight * attn_logit_softcapping

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
    
class Gemma2Attention(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 q_proj: torch.Tensor,
                 k_proj: torch.Tensor,
                 v_proj: torch.Tensor,
                 output_proj: torch.Tensor,
                 max_seq_len: int,
                 pos_embeddings: RotaryPositionalEmbeddings,
                 attn_logit_softcapping: Optional[float],
                 query_pre_attn_scalar: Optional[int],
                 kv_cache: Optional[KVCache] = None,
                 sliding_window_size: Optional[int] = None,
                 ) -> None:
        super().__init__()
        # State dimensions and head dimensions
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        assert self.num_heads % self.num_kv_heads == 0, 'The number of heads must be divisible by the number of key-value heads.'
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Projections for query, key, value and output and KV cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.kv_cache = kv_cache

        # Attention enhancements
        self.attn_logit_softcapping = attn_logit_softcapping
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.pos_embeddings = pos_embeddings
        self.sliding_window_size = sliding_window_size

        # Attention scalling
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5
    
    def forward(self, 
                x: torch.Tensor,
                *,
                mask: torch.Tensor,
                attn_type: str = 'LOCAL_SLIDING',
                input_pos: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        hidden_states_shape = x.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        if input_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({input_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # Resizing the tensors prior to ROPE
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = self.pos_embeddings(xq, input_pos=input_pos)
        xk = self.pos_embeddings(xk, input_pos=input_pos)


        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv,
                                            self.num_queries_per_kv,
                                            dim=2)
        
        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)

        # Update key-value cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, xk, xv)

        # shape: [b, 1, s, s]
        if mask is not None:
            mask = mask[:, None, :, :]

        # [batch_size, n_local_heads, input_len, max_seq_len]
        output = scaled_dot_product_attention_w_logits_softcapping(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=self.kv_cache is None and mask is None,
            scale=self.scaling,
            attn_logit_softcapping=self.attn_logit_softcapping,
            sliding_window_attn=attn_type == 'LOCAL_SLIDING',
            sliding_window_size=self.sliding_window_size)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.output_proj(output)
        return output


