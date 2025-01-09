import torch
import torch.nn as nn
from torchvision import transforms
import collections.abc
from typing import Optional, Set, Tuple, Union
import numpy as np
import random
from activations import ACT2FN
import math
from copy import deepcopy
from collections import OrderedDict, UserDict
from dataclasses import dataclass
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class DiffTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
        
        # timestep embedding 
        self.time_emb = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, timesteps):
        # timestep embedding
        t_emb = self.time_emb(timestep_embedding(timesteps, x.shape[-1]))
        t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        for attn, ff in self.layers:
            x = attn(x) + x  # add timestep embedding 
            x = ff(x + t_emb) + x  # add timestep embedding 

        return self.norm(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Timestep embedding function
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.diff_transformer = DiffTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.to_image = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h = image_height // patch_height, w = image_width // patch_width)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.encoder_unmasked = encoder.transformer
        self.encoder_masked = encoder.diff_transformer
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.enc_to_dec2 = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        # self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.decoder = DiffTransformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)

    def forward(self, img_clean, img_noise, timesteps, rand_indices=None):
        device = img_clean.device

        # get patches

        unmasked_patches = self.to_patch(img_clean)
        masked_patches = self.to_patch(img_noise)
        batch, num_patches, *_ = masked_patches.shape

        # patch to encoder tokens and add positions

        unmasked_tokens = self.patch_to_emb(unmasked_patches)
        masked_tokens = self.patch_to_emb(masked_patches)
        if self.encoder.pool == "cls":
            unmasked_tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
            masked_tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            unmasked_tokens += self.encoder.pos_embedding.to(device, dtype=unmasked_tokens.dtype) 
            masked_tokens += self.encoder.pos_embedding.to(device, dtype=masked_tokens.dtype) 

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        if rand_indices == None:
            rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        unmasked_tokens = unmasked_tokens[batch_range, unmasked_indices]
        masked_tokens = masked_tokens[batch_range, masked_indices]

        # get the patches to be masked for the final reconstruction loss
        masked_patches = masked_patches[batch_range, masked_indices]

        # attend with vision transformer
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        encoded_tokens = self.encoder_unmasked(unmasked_tokens)
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        # mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        # mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        masked_encoded_tokens = self.encoder_masked(masked_tokens, timesteps)
        masked_encoded_tokens = self.enc_to_dec(masked_encoded_tokens) + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        # decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoder_tokens[batch_range, masked_indices] = masked_encoded_tokens
        decoded_tokens = self.decoder(decoder_tokens, timesteps)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        # recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        
        outputs = torch.zeros_like(self.to_patch(img_clean))
        outputs[batch_range, unmasked_indices] = unmasked_patches[batch_range, unmasked_indices]
        outputs[batch_range, masked_indices] = pred_pixel_values.to(torch.float32)
        outputs = self.encoder.to_image(outputs)
        
        mask = torch.zeros_like(self.to_patch(img_clean))
        mask[batch_range, unmasked_indices] = 0
        mask[batch_range, masked_indices] = 1
        mask = self.encoder.to_image(mask)

        # return recon_loss, outputs
        return outputs, mask
    
if __name__ == '__main__':

    @dataclass
    class TrainingConfig:
        image_size = 224  # the generated image resolution
        train_batch_size = 16
        eval_batch_size = 16  # how many images to sample during evaluation
        num_epochs = 500
        gradient_accumulation_steps = 1
        learning_rate = 1e-4
        lr_warmup_steps = 500
        save_image_epochs = 10
        save_model_epochs = 30
        mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
        seed = 0
        num_steps = 1000
        patch_size = 16
        mask_ratio = 0.25
        hidden_size=768
        # num_hidden_layers=12
        num_enc_layers=4
        num_dec_layers=4
        num_blk_layers=3
        num_attention_heads=12
        intermediate_size=3072
        hidden_act="gelu"
        hidden_dropout_prob=0.0
        attention_probs_dropout_prob=0.0
        initializer_range=0.02
        layer_norm_eps=1e-12
        num_channels=3
        qkv_bias=True
        decoder_num_attention_heads=16
        decoder_hidden_size=512
        decoder_num_hidden_layers=8
        decoder_intermediate_size=2048
        dropout = 0.1
                
    def setup_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    config = TrainingConfig()
        
    v = ViT(
        image_size = config.image_size,
        patch_size = config.patch_size,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    mae = MAE(
        encoder = v,
        masking_ratio = 0.75,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        decoder_depth = 6       # anywhere from 1 to 8
    )

    
    img_clean = torch.randn([config.eval_batch_size, config.num_channels, config.image_size, config.image_size])
    img_noisy = torch.randn([config.eval_batch_size, config.num_channels, config.image_size, config.image_size])

    timesteps = torch.randint(0, 1000, (config.eval_batch_size,))

    loss = mae(img_clean, img_noisy, timesteps)

    pass

