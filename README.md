# Loss-Guided Layout Control

Zakaria Patel, Kirill Serkh

Paper: https://arxiv.org/abs/2405.14101

<p align="center">
  <img src="https://github.com/ZakariaPZ/loss-guided-layout-control/blob/main/diffusion_animation.gif" alt="iLGD Example"/>
</p>

### Usage
To use iLGD, you must specify the indices of the tokens in the prompt you would like to apply it to. For instance, for the prompt "a donut and a carrot", we apply iLGD on the tokens "donut" and "carrot" as follows:
      
      python3 main.py --prompt "a donut and a carrot" --indices 2 5

### Citation

      @article{patel2024enhancingimagelayoutcontrol,
          title={Enhancing Image Layout Control with Loss-Guided Diffusion Models},
          author={Zakaria Patel and Kirill Serkh},
          year={2024},
          journal={arXiv preprint arXiv:2405.14101}
      }
