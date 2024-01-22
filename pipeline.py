from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from typing import Optional, List
from injection_attention_processor import InjectionAttnProcessor, get_attention_scores
import torch
import types

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", ).to('cuda')
scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler
generator = torch.Generator(device="cuda").manual_seed(647)

# print(pipeline.scheduler.sigmas)
# print(pipeline.scheduler.timesteps)

# processor = InjectionAttnProcessor(pipeline.scheduler.sigmas, None)


for block in pipeline.unet.up_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, torch.zeros((64, 64))))

# mid block
for module in pipeline.unet.mid_block.attentions:
    for transformer_block in module.transformer_blocks:
        transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
        transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, torch.zeros((64, 64))))


for block in pipeline.unet.down_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, torch.zeros((64, 64))))

pipeline('A colorful parrot and a red hat', generator=generator).images[0].save('astronaut_rides_horse.png')


class LGDPipeline(StableDiffusionPipeline):
    pass 