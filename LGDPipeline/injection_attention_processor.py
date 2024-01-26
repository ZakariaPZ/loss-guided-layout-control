from diffusers.models.attention_processor import Attention, AttnProcessor
import torch
from typing import Optional, List, Tuple
import numpy as np
from utils import IndexTensorPair
import copy

def get_attention_scores(
    self, query: torch.Tensor, 
    key: torch.Tensor, 
    attention_mask: torch.Tensor = None,
    context_tensors: List[Tuple[int, torch.Tensor]] = None,
    injection_weight: float = 0.0,
    t = None,
    context_attention_map = None,
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
        alpha=1,
    )

    # Assuming bs of 1 (double check how the dims change)
    # Each will be of shape (nheads * bs, hidden_dim, ntokens)
    attention_scores_uncond, attention_scores_cond = attention_scores.chunk(2, dim=0)

    if context_tensors and t:
        if t < 10:
            for context_pair in context_tensors:
                context_index = context_pair.index
                context_tensor = context_pair.tensor
                model_attention_map = attention_scores_cond[:, :, context_index].clone()

                context_attention_map['resolution'] = model_attention_map.shape[1]
                # context_attention_map['map'] = model_attention_map

                context_tensor = context_tensor.flatten()
                nu_t = injection_weight * torch.max(model_attention_map)

                attention_scores_cond[:, :, context_index] += nu_t * context_tensor[None, ...]

    attention_scores_cond *= self.scale

    del baddbmm_input

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)
    
    # if t == 49 and context_tensors[0].tensor.shape[0] == 8:
    #     np.save('attention_map.npy', attention_probs.detach().cpu().numpy())

    return attention_probs


class InjectionAttnProcessor(AttnProcessor):
    def __init__(self,
                 sigma_t,
                 context_tensors: IndexTensorPair = None,
                 cross_attention_dict: dict = None,
                 nu: float = 0.75) -> None:
        self.t = 0 
        self.nu = nu
        self.sigma_t = sigma_t
        self.context_tensors = context_tensors
        self.attention_maps = cross_attention_dict

    def get_injection_scale(self):
        return self.nu * np.log(1 + self.sigma_t[self.t])
    
    def resize_context_tensor(self, attention_dim):
        resize_factor = int(64 // np.sqrt(attention_dim))
        resized_context_tensors = copy.deepcopy(self.context_tensors.copy())
        for context_pair in resized_context_tensors:
            context_pair.tensor = context_pair.tensor[0::resize_factor, 0::resize_factor].to('cuda')            
            
        return resized_context_tensors

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

        injection_weight = self.get_injection_scale() 
        self.t += 1
        resized_context_tensors = self.resize_context_tensor(query.shape[1])    

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask, resized_context_tensors, injection_weight, self.t, self.attention_maps)

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
