from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from typing import Optional, List
from injection_attention_processor import InjectionAttnProcessor, get_attention_scores
import torch
import types
from utils import IndexTensorPair
import numpy as np

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", ).to('cuda')
scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler
generator = torch.Generator(device="cuda").manual_seed(647)

pair1 = IndexTensorPair(3, torch.tensor(np.load('parrot.npy')))
pair2 = IndexTensorPair(7, torch.tensor(np.load('hat.npy')))

pairs = [pair1, pair2]
attentions = {}


# Down blocks
down_block_cross_attn_no = 0 
for block in pipeline.unet.down_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                attentions["down_block_cross_attn_" + str(down_block_cross_attn_no)] = {'resolution': None, 'map': None}
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(sigma_t=pipeline.scheduler.sigmas, context_tensors=pairs, cross_attention_dict=attentions["down_block_cross_attn_" + str(down_block_cross_attn_no)]))

                down_block_cross_attn_no += 1

# mid block
mid_block_cross_attn_no = 0 
for module in pipeline.unet.mid_block.attentions:
    for transformer_block in module.transformer_blocks:
        attentions["mid_block_cross_attn_" + str(mid_block_cross_attn_no)] = {'resolution': None, 'map': None}
        transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
        transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, pairs, cross_attention_dict=attentions["mid_block_cross_attn_" + str(mid_block_cross_attn_no)]))

        mid_block_cross_attn_no += 1


# Up blocks
up_block_cross_attn_no = 0
for block in pipeline.unet.up_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                attentions["up_block_cross_attn_" + str(up_block_cross_attn_no)] = {'resolution': None, 'map': None}
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, pairs, cross_attention_dict=attentions["up_block_cross_attn_" + str(up_block_cross_attn_no)]))

                up_block_cross_attn_no += 1

pipeline('A colorful parrot and a red hat', generator=generator).images[0].save('astronaut_rides_horse.png')

print(attentions)