from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline, EulerDiscreteScheduler, HeunDiscreteScheduler
from pipeline import LGDPipeline
from injection_attention_processor import InjectionAttnProcessor, get_attention_scores, resize_maps
import torch
import types
from utils import IndexTensorPair
import numpy as np


def run_lgd(lg_steps, injection_steps, eta, nu, bbox_corners):
    pipeline = LGDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
    scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = scheduler

    pipeline.set_params(eta=eta, lg_steps=lg_steps)

    injection_steps = injection_steps
    nu = nu

    bbox_mats = []
    for corners in bbox_corners:
        ## convert corners into bbox_array
        # mat =
        # IndexTensorPair(2, torch.tensor(np.load('plate.npy'))) 
        pass

    pair1 = ...
    pair2 = IndexTensorPair(7, torch.tensor(np.load('spoon.npy')))

    pairs = [pair1, pair2]

    token_attentions = {}
    for pair in pairs:
        token_attentions[pair.index] = []

    # Down blocks
    down_block_cross_attn_no = 0 
    for block in pipeline.unet.down_blocks:
        if 'CrossAttn' in block.__class__.__name__:
            for module in block.attentions:

                for transformer_block in module.transformer_blocks:
                    transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                    transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
                    transformer_block.attn2.set_processor(InjectionAttnProcessor(scheduler=pipeline.scheduler, context_tensors=pairs, cross_attention_dict=token_attentions, injection_steps=injection_steps, nu=nu))

                    down_block_cross_attn_no += 1

    # mid block
    mid_block_cross_attn_no = 0 
    for module in pipeline.unet.mid_block.attentions:
        for transformer_block in module.transformer_blocks:
            transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
            transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
            transformer_block.attn2.set_processor(InjectionAttnProcessor(scheduler=pipeline.scheduler, context_tensors=pairs, cross_attention_dict=token_attentions, injection_steps=injection_steps, nu=nu))

            mid_block_cross_attn_no += 1

    # Up blocks
    up_block_cross_attn_no = 0 
    for block in pipeline.unet.up_blocks:
        if 'CrossAttn' in block.__class__.__name__:
            for module in block.attentions:

                for transformer_block in module.transformer_blocks:
                    transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                    transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
                    transformer_block.attn2.set_processor(InjectionAttnProcessor(scheduler=pipeline.scheduler, context_tensors=pairs, cross_attention_dict=token_attentions, injection_steps=injection_steps, nu=nu))

                    up_block_cross_attn_no += 1

    pair3 = IndexTensorPair(2, torch.tensor(np.load('plate.npy')))
    pair4 = IndexTensorPair(7, torch.tensor(np.load('spoon.npy')))

    pairs2 = [pair3, pair4]

    pipeline.token_maps = token_attentions
    pipeline.injection_maps = pairs2

    # pipeline('A ball next to some cheese').images[0].save('astronaut_rides_horse.png')
    pipeline('A dog standing next to a stick', num_inference_steps=50).images[0].save('astronaut_rides_horse.png')

    # print(pipeline.token_maps)
    # np.save('attention_maps.npy', pipeline.token_maps)
    # print(attentions)