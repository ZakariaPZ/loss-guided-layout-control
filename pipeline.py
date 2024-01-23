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

# print(pipeline.scheduler.sigmas)
# print(pipeline.scheduler.timesteps)

# processor = InjectionAttnProcessor(pipeline.scheduler.sigmas, None)
pair1 = IndexTensorPair(3, torch.tensor(np.load('parrot.npy')))
pair2 = IndexTensorPair(7, torch.tensor(np.load('hat.npy')))
# pair1 = IndexTensorPair(1, torch.zeros((64, 64)))
# pair2 = IndexTensorPair(2, torch.zeros((64, 64)))
pairs = [pair1, pair2]

for block in pipeline.unet.up_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, pairs))

# mid block
for module in pipeline.unet.mid_block.attentions:
    for transformer_block in module.transformer_blocks:
        transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
        transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, pairs))


for block in pipeline.unet.down_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, pairs))

pipeline('A colorful parrot and a red hat', generator=generator).images[0].save('astronaut_rides_horse.png')
# pipeline('A colorful parrot and a red hat').images[0].save('astronaut_rides_horse.png')


class LGDPipeline(StableDiffusionPipeline):
    pass 