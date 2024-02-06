from diffusers import LMSDiscreteScheduler
from pipeline import LGDPipeline
from injection_attention_processor import InjectionAttnProcessor, get_attention_scores, resize_maps
import torch
import types
from utils import IndexTensorPair
import numpy as np

pipeline = LGDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to('cuda')
scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
pipeline.scheduler = scheduler
# generator = torch.Generator(device="cuda").manual_seed(647)
# generator = torch.Generator(device="cuda").manual_seed(245)

pipeline.safety_checker = None
pipeline.requires_safety_checker = False


pair1 = IndexTensorPair(3, torch.tensor(np.load('parrot.npy')))
pair2 = IndexTensorPair(7, torch.tensor(np.load('hat.npy')))

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
                transformer_block.attn2.set_processor(InjectionAttnProcessor(sigma_t=pipeline.scheduler.sigmas, context_tensors=pairs, cross_attention_dict=token_attentions))

                down_block_cross_attn_no += 1

# mid block
mid_block_cross_attn_no = 0 
for module in pipeline.unet.mid_block.attentions:
    for transformer_block in module.transformer_blocks:
        transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
        transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
        transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, pairs, cross_attention_dict=token_attentions))

        mid_block_cross_attn_no += 1

# Up blocks
up_block_cross_attn_no = 0 
for block in pipeline.unet.up_blocks:
    if 'CrossAttn' in block.__class__.__name__:
        for module in block.attentions:

            for transformer_block in module.transformer_blocks:
                transformer_block.attn2.get_attention_scores = types.MethodType(get_attention_scores, transformer_block.attn2)
                transformer_block.attn2.resize_maps = types.MethodType(resize_maps, transformer_block.attn2)
                transformer_block.attn2.set_processor(InjectionAttnProcessor(pipeline.scheduler.sigmas, pairs, cross_attention_dict=token_attentions))

                up_block_cross_attn_no += 1

pair3 = IndexTensorPair(3, torch.tensor(np.load('parrot.npy')))
pair4 = IndexTensorPair(7, torch.tensor(np.load('hat.npy')))

pairs2 = [pair3, pair4]

pipeline.token_maps = token_attentions
pipeline.injection_maps = pairs2
pipeline.lg_steps = 30

# pipeline('A cute dog with a red hat on the green grass', generator=generator).images[0].save('astronaut_rides_horse.png')
pipeline('A cute puppy with a red hat').images[0].save('astronaut_rides_horse.png')

# print(pipeline.token_maps)
# np.save('attention_maps.npy', pipeline.token_maps)
# print(attentions)