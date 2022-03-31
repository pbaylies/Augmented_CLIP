#!/usr/bin/env python
# coding: utf-8

# # Generates images from text prompts with CLIP guided diffusion.
#
# By Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings). It uses either OpenAI's 256x256 unconditional ImageNet or Katherine Crowson's fine-tuned 512x512 diffusion model (https://github.com/openai/guided-diffusion), together with CLIP (https://github.com/openai/CLIP) to connect text prompts with images.
#
# Modified by Daniel Russell (https://github.com/russelldc, https://twitter.com/danielrussruss) to include (hopefully) optimal params for quick generations in 15-100 timesteps rather than 1000, as well as more robust augmentations.
#
# **Update**: Sep 19th 2021
#
#
# Further improvements from Dango233 and nsheppard helped improve the quality of diffusion in general, and especially so for shorter runs like this notebook aims to achieve.
#
# Katherine's original notebook can be found here:
# https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA
#
# Vark has added some code to load in multiple Clip models at once, which all prompts are evaluated against, which may greatly improve accuracy.
#
# I, pbaylies, have added some code to augment a CLIP model, which may also improve the results.

import torch
# Check the GPU status
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

import os
model_path = os.getcwd() + '/'
try:
    os.mkdir(model_path + 'image_storage')
except:
    pass

# Get which pip command to use
import subprocess
pip_cmd = subprocess.check_output(['which', 'pip']).decode('utf-8').strip()
# If pip is not installed, check for pip3
if not os.path.isfile(pip_cmd):
    pip_cmd = subprocess.check_output(['which', 'pip3']).decode('utf-8').strip()
    if not os.path.isfile(pip_cmd):
        print('Please install pip3 or pip')
        exit()

def install_dependencies():
    os.system('git clone https://github.com/openai/CLIP')
    os.system('git clone https://github.com/crowsonkb/guided-diffusion')
    os.system('git clone https://github.com/leibovit/Sparse-Linear-Networks')
    os.system(pip_cmd + ' install -e ./CLIP')
    os.system(pip_cmd + ' install -e ./guided-diffusion')
    os.system(pip_cmd + ' install lpips datetime')

# If dependencies are not installed, install them
if not os.path.isdir(model_path + 'Sparse-Linear-Networks'):
    install_dependencies()

import gc
import io
import math
import sys
import lpips
from PIL import Image, ImageOps, ImageFile
import requests
import torch
from torch import nn
from torchvision.transforms import Compose
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm.notebook import tqdm
sys.path.append('./CLIP')
sys.path.append('./guided-diffusion')
# Butterfly Layer
sys.path.append('./Sparse-Linear-Networks')
import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from butterfly import Butterfly
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

# https://github.com/pratogab/batch-transforms
class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        if (not torch.is_tensor(tensor)):
            tensor = torch.tensor(tensor)
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor

from abc import ABC, abstractmethod

class BaseFeatureModel(ABC):
    # Get the model name at initialization time
    def __init__(self, name, device):
        self.name = name
        self.device = device
        super().__init__()

    # Return dimension of features returned by the model
    @property
    @abstractmethod
    def size(self):
        pass

    # Return expected image input size used by the model
    @property
    @abstractmethod
    def input_size(self):
        pass


    # Perform inference on an image, return features
    @abstractmethod
    def run(self, image):
        pass

class CLIPFeatureModel(BaseFeatureModel):
    def __init__(self, name, device):
        super().__init__(name, device)
        # Initialize the model
        self.model, _ = clip.load(self.name, jit=False)
        self.model = self.model.eval().requires_grad_(False).to(device)
        self.transform = Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), inplace=True, device=device),
        ])

        # Feature embedding size and input size of currently released CLIP models computed below
        self.input_size = (224,224)
        if self.name == "RN50":
            self.size = 1024
        elif self.name == "RN50x4":
            self.size = 640
            self.input_size = (288,288)
        elif self.name == "RN50x16":
            self.size = 768
            self.input_size = (384,384)
        elif self.name == "RN50x64":
            self.size = 1024
            self.input_size = (448,448)
        elif self.name == "ViT-L/14":
            self.size = 768
        else:
            self.size = 512

    def size(self):
        return self.size

    def input_size(self):
        return self.input_size

    def run(self, image):
        image = self.transform(image).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(image)

    def encode_text(self, text):
        with torch.no_grad():
            text = clip.tokenize(text, truncate=True).to(self.device)
            return self.model.encode_text(text)

    def logits_per_image(self, image, text):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, text)
            return logits_per_image

    def softmax(self, image, text):
        logits_per_image = self.logits_per_image(image, text)
        return logits_per_image.softmax(dim=-1).cpu().numpy()

def fetch_diffusion_model(diffusion_model):
    if diffusion_model == '256x256_diffusion_uncond':
        os.system("wget --continue 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt' -P " + model_path)
    elif diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        os.system("wget --continue 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt' -P " + model_path)

## Define necessary functions
# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)

def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)

def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_x, side_y), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0]//3, out.shape[1])
        out = TF.resize(size=(side_x, side_y), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            #T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2]//4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn//4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1,).normal_(mean=.8, std=.3).clip(float(self.cut_size/max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)

def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0, rand=0.02):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        # add stochastic gradient clipping
        #clipped_grad = g_data * (((max_norm * torch.rand_like(g_data)).clamp_(min=1e-6)) / grad_norm.clamp(min=1e-6))
        # add noise
        #clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6)) + (max_norm / 50.0) * torch.randn_like(g_data)
        # do both?
        #clipped_grad = g_data * (((max_norm * torch.rand_like(g_data)).clamp_(min=1e-6)) / grad_norm.clamp(min=1e-6)) + (max_norm / 100.0) * torch.randn_like(g_data)

        if rand > 0.0:
            clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6)) + (rand*max_norm) * torch.randn_like(g_data)
        else:
            clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))

        #clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        #new_grads = torch.where(grad_norm < max_norm, g_data + (max_norm / 30.0) * torch.randn_like(g_data), clipped_grad)
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)

import click
@click.command()
@click.option('--verbose', is_flag=True, default=True, help='Print more information.')
@click.option('--diffusion_model', default='256x256_diffusion_uncond', help='Diffusion model to use. Options: 256x256_diffusion_uncond, 512x512_diffusion_uncond_finetune_008100')
@click.option('--use_clip_models', default=['ViT-B/32','ViT-B/16','RN101'], multiple=True, help='CLIP models to use.')
@click.option('--timestep_respacing', default='ddim100', help='Modify this value to change the number of timesteps.')
@click.option('--diffusion_steps', default=1000, help='Number of timesteps to run diffusion for.')
@click.option('--side_x', default=0, help='Width of image to generate.')
@click.option('--side_y', default=0, help='Height of image to generate.')
@click.option('--seed', default=-1, help='Seed for random number generator.')
@click.option('--clip_denoised', default=True, help='Whether to use denoised images for CLIP.')
@click.option('--fuzzy_prompt', default=False, help='Whether to add multiple noisy prompts to the prompt losses.')
@click.option('--rand_mag', default=1.0, help='Magnitude of random noise to add to the image.')
@click.option('--eta', default=0.0, help='DDIM hyperparameter.')
@click.option('--init_image', '-init', default=None, help='Path to an image to use as the initial image.')
@click.option('--init_scale', default=0, help='Scale of the initial image.')
@click.option('--skip_timesteps', '-s', default=0, help='Number of timesteps to skip.')
@click.option('--perlin_init', default=False, help='Whether to start with random perlin noise.')
@click.option('--perlin_mode', default='mixed', help='Whether to use grayscale or color perlin noise.')
@click.option('--skip_augs', default=False, help='Whether to skip torchvision augmentations.')
@click.option('--randomize_class', default=True, help='Whether to randomly change the imagenet class.')
@click.option('--clamp_grad', default=True, help='Whether to use adaptive clip grad in the cond_fn.')
@click.option('--clip_guidance_scale', default=25000, help='How much the image should look like the prompt.')
@click.option('--tv_scale', default=150, help='How smooth the final output should be.')
@click.option('--range_scale', default=150, help='How far out of range RGB values are allowed to be.')
@click.option('--sat_scale', default=0, help='How much saturation is allowed.')
@click.option('--cutn', '-c', default=16, help='How many crops to take from the image.')
@click.option('--cutn_batches', '-cb', default=2, help='How many batches to accumulate CLIP gradient from.')
@click.option('--use_aug_clip', default=True, help='Whether to use Aug CLIP models to augment ViT-B/32 CLIP embeddings.')
@click.option('--aug_clip_models', multiple=True, default=['_avg1','_avgxs','_doddbf1','_dallbf1'], help='List of Aug CLIP models to use.')
@click.option('--aug_model_path', default='checkpoints/', help='Path to the models to use for Aug CLIP models.')
@click.option('--n_batches', '-n', default=6, help='Number of batches to generate.')
@click.option('--pick_best', '-pb', default=3, help='Pick samples with the best loss to continue generating variations.')
@click.option('--batch_size', '-b', default=1, help='Number of images to generate in a batch.')
@click.option('--text_prompts', '-p', multiple=True, default=["Cyberpunk sorceress"], help='Prompts to use.')
@click.option('--image_prompts', '-i', default=[], multiple=True, help='Image prompts to use.')
@click.option('--loop', '-l', default=False, help='Whether to keep generating images in a loop.')
def main(verbose, diffusion_model, use_clip_models, timestep_respacing, diffusion_steps, side_x, side_y, seed, clip_denoised, fuzzy_prompt, rand_mag, eta, init_image, init_scale, skip_timesteps, perlin_init, perlin_mode, skip_augs, randomize_class, clamp_grad, clip_guidance_scale, tv_scale, range_scale, sat_scale, cutn, cutn_batches, use_aug_clip, aug_clip_models, aug_model_path, n_batches, pick_best, batch_size, text_prompts, image_prompts, loop):
    aug_models = []
    if use_aug_clip:
        aug_models = aug_clip_models
        num_models = len(aug_models)
        model_depth = 100
        model_avg_freq = 10
        bf_models = ["_davgbf1","_doddbf1","_devenbf1","_dallbf1"]
    else:
        num_models = 0
        model_depth = 0
        model_avg_freq = 0
        bf_models = []

    if seed < 0:
        seed = random.randint(0, 2**32) # Choose a random seed if one isn't specified
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if verbose:
        print('Using seed: {}'.format(seed))

    clip_models = []
    fetch_diffusion_model(diffusion_model)
    def do_run():
        nonlocal clip_guidance_scale
        normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        global cur_pct
        loss_values = []
        final_values = []
        images = []
        target_embeds, weights = [], []
        model_stats = []
        first = use_aug_clip
        for clip_model in clip_models:
            model_stat = {"clip_model":None,"target_embeds":[],"make_cutouts":None,"weights":[]}
            model_stat["clip_model"] = clip_model
            model_stat["make_cutouts"] = MakeCutouts(clip_model.model.visual.input_resolution, cutn, skip_augs=skip_augs)
            for prompt in text_prompts:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(prompt).float()
                # normalize and weighted average with augmented embeddings below
                with torch.no_grad():
                    orig_txt = txt.clone()
                    if first:
                        txts = []
                        for i in range(0, num_models):
                            txt = orig_txt.clone()
                            for j in range(0, model_depth):
                                first_txt = txt.clone()
                                (std1, mean1) = torch.std_mean(txt)
                                txt = text_to_images[i](txt)
                                (std2, mean2) = torch.std_mean(txt)
                                txt = mean1+std1*((txt-mean2)/(std2))
                                txt0 = txt.clone()
                                (std1, mean1) = torch.std_mean(txt)
                                txt = image_to_texts[i](txt)
                                (std2, mean2) = torch.std_mean(txt)
                                txt = mean1+std1*((txt-mean2)/(std2))
                                txt = 0.25*orig_txt+0.25*first_txt+0.25*txt+0.25*txt0
                                if (random.randint(0, model_avg_freq) == 0 and len(txts) > 0):
                                    txt = 0.5*txt + txts[-random.randint(0, j)]
                                txts.append(txt.clone())
                    else:
                        txts = [orig_txt]

                if fuzzy_prompt:
                    for i in range(25):
                        for txt in txts:
                            model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * rand_mag).clamp(0,1))
                            model_stat["weights"].append(weight)
                else:
                    for txt in txts:
                        model_stat["target_embeds"].append(txt)
                        model_stat["weights"].append(weight)

            for prompt in image_prompts:
                path, weight = parse_prompt(prompt)
                img = Image.open(fetch(path)).convert('RGB')
                img = TF.resize(img, min(side_x, side_y, *img.size), Image.LANCZOS)
                batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                embed = clip_model.model.encode_image(normalize(batch)).float()
                if fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0,1))
                        weights.extend([weight / cutn] * cutn)
                else:
                    model_stat["target_embeds"].append(embed)
                    model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)
            first = False

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert('RGB')
            init = init.resize((side_x, side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if perlin_init:
            if perlin_mode == 'color':
                init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, False)
            elif perlin_mode == 'gray':
                init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, True)
                init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)
            else:
                init = create_perlin_noise([1.5**-i*0.5 for i in range(12)], 1, 1, False)
                init2 = create_perlin_noise([1.5**-i*0.5 for i in range(8)], 4, 4, True)

            init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
            del init2

        cur_t = None

        def cond_fn(x, t, y=None):
            nonlocal clip_guidance_scale
            global cur_pct
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                n = x.shape[0]
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'y': y})
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

                for model_stat in model_stats:
                    for i in range(cutn_batches):
                        clip_in = normalize(model_stat["make_cutouts"](x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].model.encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([cutn, n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item()) # log loss, probably shouldn't do per cutn_batch
                        if (loss_values[-1] != loss_values[-1]):
                            clip_guidance_scale /= 2
                            return 0
                        x_in_grad += (torch.autograd.grad(losses.sum() * (clip_guidance_scale + clip_guidance_scale * (1-abs(2*(cur_pct-0.5)))), x_in)[0] / cutn_batches)
                tv_losses = tv_loss(x_in)
                range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1,max=1)).mean()
                loss = tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + sat_losses.sum() * sat_scale
                if init is not None and init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            if clamp_grad:
                adaptive_clip_grad([x], rand=1.0)
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=0.05) / magnitude
            return grad

        if model_config['timestep_respacing'].startswith('ddim'):
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.p_sample_loop_progressive

        i = 0
        while i < n_batches:
            cur_t = diffusion.num_timesteps - skip_timesteps - 1
            cur_pct = cur_t / diffusion.num_timesteps
            if model_config['timestep_respacing'].startswith('ddim'):
                samples = sample_fn(
                    model,
                    (batch_size, 3, side_y, side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_timesteps,
                    init_image=init,
                    randomize_class=randomize_class,
                    eta=eta,
                )
            else:
                samples = sample_fn(
                    model,
                    (batch_size, 3, side_y, side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_timesteps,
                    init_image=init,
                    randomize_class=randomize_class,
                )
            for j, sample in enumerate(samples):
                if (loss_values[-1] != loss_values[-1]):
                    break
                cur_t -= 1
                if cur_t == -1:
                    for k, image in enumerate(sample['pred_xstart']):
                        current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
                        filename = f'progress_batch{i:05}_iteration{j:05}_output{k:05}_{current_time}.png'
                        image -= image.min()
                        image /= image.max()
                        image = TF.to_pil_image(image)
                        fpath = model_path + 'image_storage/' + filename
                        image.save(fpath)
                        final_values.append(np.array(loss_values[-8:]).mean())
                        images.append(fpath)
                        clip_guidance_scale *= 1.1
            i+=1

            plt.plot(np.array(loss_values), 'r')
        return (final_values, images)


    ## Load Diffusion and CLIP models

    model_config = model_and_diffusion_defaults()
    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': diffusion_steps,
            'rescale_timesteps': True,
            'timestep_respacing': timestep_respacing,
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })
    elif diffusion_model == '256x256_diffusion_uncond':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': diffusion_steps,
            'rescale_timesteps': True,
            'timestep_respacing': timestep_respacing,
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        })
    if side_x == 0:
        side_x = model_config['image_size']
    if side_y == 0:
        side_y = model_config['image_size']

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(model_path + diffusion_model + '.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()

    for model_name in use_clip_models:
        if verbose:
            print('Initializing CLIP model %s' % model_name)
        clip_models.append(CLIPFeatureModel(model_name, device))
    lpips_model = None
    #lpips_model = lpips.LPIPS(net='vgg').to(device)
    #AlexNet model also works fine for less RAM usage
    if init_image is not None and init_scale:
        lpips_model = lpips.LPIPS(net='alex').to(device)

    # Definition of BF
    def BF(input_dim,output_dim):
        n1,n2 = input_dim, output_dim
        k1 = 256
        k2 = 256
        first_gadget = Butterfly(in_size=n1, out_size=k1)
        second_gadget = nn.Linear(k1,k2)
        third_gadget = Butterfly(in_size=k2, out_size= n2)
        return nn.Sequential(first_gadget,second_gadget,third_gadget)

    def get_bf_model(in_size = 512, out_size = 512):
        return nn.Sequential(
            BF(in_size, in_size),
            nn.ELU(),
            BF(in_size, in_size),
            nn.ELU(),
            BF(in_size, in_size),
            nn.ELU(),
            BF(in_size, out_size),
            nn.Tanh()
        )

    def load_bf_model(filename, in_size = 512, out_size = 512):
        model = get_bf_model(in_size, out_size)
        model.load_state_dict(torch.load(filename))
        return model

    ## Diffuse!

    gc.collect()
    torch.cuda.empty_cache()
    best_losses = []
    best_images = []
    best_images_log = []
    ls_run = []
    im_run = []
    model_used_log = []

    start = use_aug_clip
    try:
        while True:
            if start:
                text_to_images = []
                image_to_texts = []
                pick_models = np.random.choice(aug_models, size=[num_models], replace=False)
                print(pick_models)
                model_used_log.append(pick_models)
                for pm in pick_models:
                    if pm in bf_models:
                        text_to_image = load_bf_model(aug_model_path + "t2i" + pm + ".pt")
                    else:
                        text_to_image = torch.load(aug_model_path + "t2i" + pm + ".pt")
                    text_to_image.requires_grad_(False).eval().to(device)
                    if pm in bf_models:
                        image_to_text = load_bf_model(aug_model_path + "t2i" + pm + ".pt")
                    else:
                        image_to_text = torch.load(aug_model_path + "i2t" + pm + ".pt")
                    image_to_text.requires_grad_(False).eval().to(device)
                    text_to_images.append(text_to_image)
                    image_to_texts.append(image_to_text)
                start = False
            (ls_run, im_run) = do_run()
            if not loop:
                break
            best_losses = best_losses + ls_run
            best_images = best_images + im_run
            init_scale = 0
            skip_timesteps = random.randint(6*diffusion.num_timesteps//25, 15*diffusion.num_timesteps//25)
            seed = random.randint(0, 2**32)
            best_indexes = np.argpartition(np.array(best_losses), kth=(pick_best-1))[0:pick_best]
            best_losses = list(np.array(best_losses)[best_indexes][0:pick_best])
            best_images = list(np.array(best_images)[best_indexes][0:pick_best])
            init_image = best_images[random.randint(0, pick_best-1)]
            best_images_log.append(init_image)
    except KeyboardInterrupt:
        pass
    finally:
        print('seed', seed)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
