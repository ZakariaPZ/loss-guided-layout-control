import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple
from utils import IndexTensorPair
import copy

from diffusers.models.attention_processor import Attention, AttnProcessor



def resize_maps(self, 
                weight, 
                dims=(16, 16)):
    
    return torch.nn.functional.interpolate(weight.unsqueeze(0), dims, mode='bilinear').squeeze()

def get_attention_scores(
    self, 
    query: torch.Tensor, 
    key: torch.Tensor, 
    attention_mask: torch.Tensor = None,
    token_injection_tensors: List[Tuple[int, torch.Tensor]] = None,
    injection_weight: float = 0.0,
    t = None,
    context_attention_map = None,
    injection_steps = None
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

    ### Inject attention
    # Only need the conditional attention maps
    attention_scores_uncond, attention_scores_cond = attention_scores.chunk(2, dim=0)

    # Only inject attention for the first few steps to bias the latents
    if token_injection_tensors and (t > 0 and t < injection_steps):
        
        # required to create a view with requires_grad=True
        conditional_scores = attention_scores_cond.clone() 

        for token_injection_tensor in token_injection_tensors:

            token_idx = token_injection_tensor.idx
            injection_tensor = token_injection_tensor.tensor.clone()
            model_attention_map = conditional_scores[:, :, token_idx].clone()

            injection_tensor = injection_tensor.flatten()

            # injection_weight * max(QK^T)
            nu_t = injection_weight * torch.max(model_attention_map) 

            # inject the attention into the appriopriate token's attention map
            conditional_scores[:, :, token_idx] += nu_t * injection_tensor[None, ...]
        
        # recombine the attention scores
        attention_scores = torch.cat((attention_scores_uncond, conditional_scores), dim=0)
    ### End injection
        
    attention_scores *= self.scale

    del baddbmm_input

    if self.upcast_softmax:
        attention_scores = attention_scores.float()

    attention_probs = attention_scores.softmax(dim=-1)
    del attention_scores

    attention_probs = attention_probs.to(dtype)

    ### Extract attention maps to attention store (used in pipeline for loss-guidance)
    # Only need the conditional attention maps
    attention_probs_uncond, attention_probs_cond = attention_probs.chunk(2, dim=0)

    # Record the attention maps in the attention store so that we can compute the loss 
    # between the attention maps and the target attention maps for loss-guidance
    if token_injection_tensors and t > 0:

        for token_injection_tensor in token_injection_tensors:
            token_idx = token_injection_tensor.idx
            model_attention_map = attention_probs_cond[:, :, token_idx].clone()

            nheads = model_attention_map.shape[0]
            attention_map_dim = int(np.sqrt(model_attention_map.shape[1]))

            resized_model_attention_map = self.resize_maps(model_attention_map.view((nheads, attention_map_dim, attention_map_dim)))
            resized_model_attention_map = resized_model_attention_map.view((nheads, -1))
            context_attention_map[token_idx].append(resized_model_attention_map)
    ### End attention extraction
            
    return attention_probs


class InjectionAttnProcessor(AttnProcessor):
    def __init__(self,
                 scheduler: object = None,
                 injection_steps: int = None,
                 token_injection_tensors: IndexTensorPair = None,
                 attn_store: dict = None,
                 nu: float = 0.0) -> None:
        
        self.injection_steps = injection_steps
        self.nu = nu
        self.scheduler = scheduler
        self.token_injection_tensors = token_injection_tensors
        self.attn_store = attn_store

    def get_injection_scale(self):
        return self.nu * np.log(1 + self.scheduler.sigmas[self.scheduler.step_index].cpu().numpy())
    
    def resize_injection_tensors(self, attention_dim):
        resize_factor = int(64 // np.sqrt(attention_dim))
        token_injection_tensors = copy.deepcopy(self.token_injection_tensors) 

        for token_injection_tensor in token_injection_tensors:
            
            token_injection_tensor.tensor = token_injection_tensor.tensor[0::resize_factor, 0::resize_factor]           
            
        return token_injection_tensors

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
        attn_dim = query.shape[1]
        token_injection_tensors = self.resize_injection_tensors(attn_dim)    

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, 
                                                    key, 
                                                    attention_mask, 
                                                    token_injection_tensors, 
                                                    injection_weight, 
                                                    self.scheduler.step_index, 
                                                    self.attn_store,
                                                    self.injection_steps)

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
