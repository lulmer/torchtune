import torch
import torch.nn as nn
from .gemma_attention import Gemma2Attention

class Gemma2DecoderLayer(nn.Module):
    def __init__(self, 
                 attn : Gemma2Attention, 
                 mlp : nn.Module, 
                 input_layernorm: nn.Module, 
                 post_attn_layernorm: nn.Module,
                 pre_ffn_layernorm: nn.Module,
                 post_ffn_layernorm: nn.Module):
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm
        self.pre_ffn_layernorm = pre_ffn_layernorm
        self.post_ffn_layernorm = post_ffn_layernorm
        
    def forward(self, hidden_states, *, mask, attn_type, input_pos):
        
        # Attention Block
        y = self.input_layernorm(hidden_states)
        y = self.attn(y, attn_type=attn_type, mask=mask, input_pos=input_pos)
        y = self.post_attn_layernorm(y + hidden_states)

        # FFN Block
        residual = y
        if self.pre_ffn_layernorm is not None:
            y = self.pre_ffn_layernorm(y)
        y = self.mlp(y)
        if self.post_attn_layernorm is not None:
            y = self.post_ffn_layernorm(y)
        y = y + residual
        
        return y


