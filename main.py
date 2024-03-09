import torch
import numpy as np

import types
from typing import List, Tuple
from utils import IndexTensorPair

from diffusers import LMSDiscreteScheduler
from LGDPipeline.pipeline import LGDPipeline
from attention.injection_attention_processor import InjectionAttnProcessor, get_attention_scores, resize_maps
import argparse


def run_lgd(bbox_corners: List[Tuple[int, int, int, int]], 
            indices: List[int],
            prompt: str,
            lg_steps: int = 25,
            injection_steps: int = 11, 
            eta: float = 8/25, 
            nu: float = 0.75):
    
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

    pipeline(prompt, num_inference_steps=50).images[0].save(f'{prompt}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LGD')
    parser.add_argument('--bbox_corners', type=int, nargs='+', help='List of bounding box corners')
    parser.add_argument('--indices', type=int, nargs='+', help='List containing the indices associating prompt tokens \
                                                                with bounding boxes. The indices should be in the same \
                                                                order as the bounding boxes.')
    parser.add_argument('--lg_steps', type=int, default=25, help='Number of loss-guidance steps')
    parser.add_argument('--injection_steps', type=int, default=11, help='Number of injection steps')
    parser.add_argument('--eta', type=float, default=8/25, help='Loss-guidance value')
    parser.add_argument('--nu', type=float, default=0.75, help='Injection strength')
    parser.add_argument('--prompt', type=str, help='The Stable Diffusion prompt to use.')

    args = parser.parse_args()

    if len(args.bbox_corners) is None:
        raise ValueError('You must provide the bounding corners as [bottom_left_x, bottom_left_y, width, height].')
    elif len(args.bbox_corners) % 4 != 0:
        raise ValueError('The number of corners is invalid. You must provide the four corners\
                          as [bottom_left_x, bottom_left_y, width, height].')

    if prompt is None:
        raise ValueError('You must provide a prompt.')

    run_lgd(bbox_corners=args.bbox_corners, 
            indices=args.indices, 
            prompt=args.prompt,
            lg_steps=args.lg_steps, 
            injection_steps=args.injection_steps, 
            eta=args.eta, 
            nu=args.nu)
    