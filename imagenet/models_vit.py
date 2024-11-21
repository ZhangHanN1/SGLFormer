from sglformer_ima import sglformer_ima
from sglformer_ima2 import sglformer_ima2
import torch.nn as nn
from functools import partial


def sglformer_ima_8_384(T=4, **kwargs):
    model = sglformer_ima(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

def sglformer_ima2_8_384(T=4, **kwargs):
    model = sglformer_ima2(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=6, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

def sglformer_ima_8_512(T=4, **kwargs):
    model = sglformer_ima(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

def sglformer_ima2_8_512(T=4, **kwargs):
    model = sglformer_ima2(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=512, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

def sglformer_ima_8_768(T=4, **kwargs):
    model = sglformer_ima(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

def sglformer_ima2_8_768(T=4, **kwargs):
    model = sglformer_ima2(
        T=T,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=8, sr_ratios=1,
        **kwargs
    )
    return model

