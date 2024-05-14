import torch
import numpy as np

import types
from typing import List, Tuple
from utils import IndexTensorPair
import ast
import argparse
import os

from diffusers import LMSDiscreteScheduler
from LGDPipeline.pipeline import LGDPipeline
from attention.injection_attention_processor import InjectionAttnProcessor, get_attention_scores, resize_maps


def run_lgd(bbox_corners: List[Tuple[int, int, int, int]], 
            indices: List[int],
            prompt: str,
            lg_steps: int = 25,
            injection_steps: int = 11, 
            eta: float = 12/25, 
            nu: float = 0.75,
            seed: int = 0):
    
    pipeline = LGDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
    scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = scheduler

    pipeline.set_ilgd_params(eta=eta, lg_steps=lg_steps)
    injection_steps = injection_steps
    nu = nu

    attn_store = {}
    token_injection_tensors = []
    token_lg_tensors = []
    for idx, corners in zip(indices, bbox_corners):
        # convert corners to a binary mask
        x, y, w, h = corners
        mat = np.zeros((64, 64))
        mat[y:y+h+1, x:x+w+1] = 1
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

    if not os.path.exists('images'):
        os.makedirs('images')

    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipeline(prompt, num_inference_steps=50, generator=generator, guidance_scale=7.5).images[0].save(f'images/{prompt}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LGD')
    parser.add_argument('--bbox_corners', type=ast.literal_eval, nargs='+', help='List of bounding box corners')
    parser.add_argument('--indices', type=ast.literal_eval, nargs='+', help='List containing the indices associating prompt tokens \
                                                                with bounding boxes. The indices should be in the same \
                                                                order as the bounding boxes.')
    parser.add_argument('--lg_steps', type=int, default=25, help='Number of loss-guidance steps')
    parser.add_argument('--injection_steps', type=int, default=11, help='Number of injection steps')
    parser.add_argument('--eta', type=float, default=12/25, help='Loss-guidance value')
    parser.add_argument('--nu', type=float, default=0.75, help='Injection strength')
    parser.add_argument('--prompt', type=str, help='The Stable Diffusion prompt to use.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()
    bbox_corners = args.bbox_corners[0]
    indices = args.indices[0]
    seed = args.seed

    if bbox_corners is None:
        raise ValueError('You must provide the bounding corners as [bottom_left_x, bottom_left_y, width, height].')

    if args.prompt is None:
        raise ValueError('You must provide a prompt.')

    print(args.nu)
    print(args.eta)

    run_lgd(bbox_corners=bbox_corners, 
            indices=indices, 
            prompt=args.prompt,
            lg_steps=args.lg_steps, 
            injection_steps=args.injection_steps, 
            eta=args.eta, 
            nu=args.nu,
            seed=seed)