import torch.nn as nn
import torch

# Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
class Gemma2RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self, dim=768, max_seq_len=2048, base=10000, device=None
    ):
        super().__init__()
        self.dim = dim
        self.rope_base = base
        self.device = device
        self.max_position_embeddings = max_seq_len
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
    
    @torch.no_grad()
    def forward(self, x, *, input_pos, seq_len=None):
        if seq_len is None :
            seq_len = x.shape[1] # self.max_position_embeddings
        if input_pos is None : 
            input_pos = torch.arange(seq_len).unsqueeze(0).expand(x.shape[0], seq_len)
        # input comming in : torch.randn(bsz, seq_len, num_heads, head_dim)
        print("x shape : ",x.shape)
        print("input_pos shape : ",input_pos.shape)
        x = torch.transpose(x, 1, 2)
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(input_pos.shape[0], -1, 1)
        position_ids_expanded = input_pos[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        return torch.transpose(self._apply_rotary_pos_emb(x, cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)),1,2)
    
    def _rotate_half(self,x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _apply_rotary_pos_emb(self, x, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.
        Args:
            x (`torch.Tensor`): The input tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        x_new = (x * cos) + (self._rotate_half(x) * sin)

        return x_new