from diffusers.models.attention_processor import Attention, AttnProcessor
import torch
from typing import Optional, List
import numpy as np
from pprint import pprint

def get_attention_scores(
    self, query: torch.Tensor, 
    key: torch.Tensor, 
    attention_mask: torch.Tensor = None,
    context_tensor: torch.Tensor = None
) -> torch.Tensor:
    r"""
    Compute the attention scores.

    Args:
        query (`torch.Tensor`): The query tensor.
        key (`torch.Tensor`): The key tensor.
        attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    Returns:
        `torch.Tensor`: The attention probabilities/scores.
    """
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=self.scale,
    )

    # Assuming bs of 1 (double check how the dims change)
    # Each will be of shape (nheads * bs, hidden_dim, ntokens)
    attention_scores_cond, attention_scores_uncond = attention_scores.chunk(2, dim=0)
    # print('Conditional dims: ', attention_scores_cond.shape)
    # print('Unconditional dims: ', attention_scores_uncond.shape)
    # print('Total: ', attention_scores.shape)

    # attention_scores_cond += context_tensor

    del baddbmm_input

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    return attention_probs


class InjectionAttnProcessor(AttnProcessor):
    def __init__(self,
                 sigma_t, 
                 context_tensor: torch.Tensor) -> None:
        super().__init__()
        self.t = 0 
        self.sigma_t = sigma_t
        self.context_tensor = context_tensor
        self.attention_map = None

    def get_weight_scale(self, t):
        return 0.75 * np.log(1 + self.sigma_t[t])
    
    def resize_context_tensor(self, attention_dim):
        resize_factor = int(64 // np.sqrt(attention_dim))
        return self.context_tensor[0::resize_factor, 0::resize_factor]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        
    ) -> torch.Tensor:
        """
        Copied heavily from https://github.com/huggingface/diffusers/blob/ac61eefc9f2fbd4d2190d5673a4fcd77da9a93ab/src/diffusers/models/attention_processor.py. 
        """

        residual = hidden_states

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)
        weight_scale = self.get_weight_scale(self.t)
        self.t += 1
        resized_context_tensor = self.resize_context_tensor(query.shape[1])    
        print('Resized context tensor: ', resized_context_tensor.shape)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
