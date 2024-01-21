from diffusers import StableDiffusionPipeline
from typing import Optional, List
from injection_attention_processor import InjectionAttnProcessor, get_attention_scores
import torch
import types

model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
generator = torch.Generator(device="cuda").manual_seed(647)

processor = InjectionAttnProcessor(0.1, 0.1)

for block in model.unet.up_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(0.1, 0.1))

for block in model.unet.down_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(0.1, 0.1))


model('A colorful parrot and a red hat', generator=generator).images[0].save('astronaut_rides_horse.png')


class LGDPipeline(StableDiffusionPipeline):
    pass 