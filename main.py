from diffusers import LMSDiscreteScheduler
from LGDPipeline.pipeline import LGDPipeline
from attention.injection_attention_processor import InjectionAttnProcessor, get_attention_scores, resize_maps
import torch
import types
from utils import IndexTensorPair
import numpy as np


def run_lgd(lg_steps, injection_steps, eta, nu, bbox_corners, indices):
    pipeline = LGDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
    scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = scheduler

    pipeline.set_params(eta=eta, lg_steps=lg_steps)
    injection_steps = injection_steps
    nu = nu

    attn_store = {}
    token_injection_tensors = []
    token_lg_tensors = []
    for idx, corners in zip(indices, bbox_corners):
        # convert corners to a binary mask
        x, y, w, h = corners
        mat = np.zeros((64, 64))
        mat[y:y+h, x:x+w] = 1
        bbox_mat = IndexTensorPair(idx, torch.tensor(mat))
        token_injection_tensors.append(bbox_mat)
        token_lg_tensors.append(bbox_mat)

        # initialize the attention store for each token
        attn_store[idx] = []

    # Down blocks
    for block in pipeline.unet.down_blocks:
        if 'CrossAttn' in block.__class__.__name__:
            for module in block.attentions:

                for transformer_block in module.transformer_blocks:
                    transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                    transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
                    transformer_block.attn2.set_processor(InjectionAttnProcessor(scheduler=pipeline.scheduler, 
                                                                                token_injection_tensors=token_injection_tensors, 
                                                                                attn_store=attn_store, 
                                                                                injection_steps=injection_steps, 
                                                                                nu=nu))

    # mid block
    for module in pipeline.unet.mid_block.attentions:
        for transformer_block in module.transformer_blocks:
            transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
            transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
            transformer_block.attn2.set_processor(InjectionAttnProcessor(scheduler=pipeline.scheduler, 
                                                                        token_injection_tensors=token_injection_tensors, 
                                                                        attn_store=attn_store, 
                                                                        injection_steps=injection_steps, 
                                                                        nu=nu))

    # Up blocks
    for block in pipeline.unet.up_blocks:
        if 'CrossAttn' in block.__class__.__name__:
            for module in block.attentions:

                for transformer_block in module.transformer_blocks:
                    transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                    transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
                    transformer_block.attn2.set_processor(InjectionAttnProcessor(scheduler=pipeline.scheduler, 
                                                                                token_injection_tensors=token_injection_tensors, 
                                                                                attn_store=attn_store, 
                                                                                injection_steps=injection_steps, 
                                                                                nu=nu))


    pipeline.attn_store = attn_store
    pipeline.target_attn_maps = token_lg_tensors

    pipeline('A dog standing next to a stick', num_inference_steps=50).images[0].save('astronaut_rides_horse.png')
