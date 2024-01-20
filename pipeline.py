from diffusers import StableDiffusionPipeline
from typing import Optional, List
from injection_attention_processor import get_attention_scores, attention_processor
import torch
import types

model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
generator = torch.Generator(device="cuda").manual_seed(647)


for block in model.unet.up_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.processor.__class__.__call__ = attention_processor

for block in model.unet.down_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.processor.__class__.__call__ = attention_processor



model('A colorful parrot and a red hat', generator=generator).images[0].save('astronaut_rides_horse.png')


class LGDPipeline(StableDiffusionPipeline):
    pass 