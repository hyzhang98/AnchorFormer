import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import timm.models.vision_transformer as timm_vit

from model.AnchorViT import AnchorAttention
from timm.layers import Mlp, DropPath
from typing import Optional

__all__ = [
    'anchor_deit_tiny_distilled_patch16_224',
    'anchor_deit_small_distilled_patch16_224',
    'anchor_deit_base_distilled_patch16_224',
    'anchor_deit_base_distilled_patch16_384',
]

class AnchorBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            proj_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            anchors: int=30,
            sparse: int=5
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AnchorAttention(
            dim,
            anchors=anchors,
            heads=num_heads,
            dim_head=dim // num_heads,
            dropout=proj_drop,
            k_sparse=sparse
        )
        self.ls1 = timm_vit.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = timm_vit.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class DistilledAnchorViT(VisionTransformer):
    def __init__(self, anchors=30, sparse=5, block_fn = AnchorBlock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], kwargs['depth'])] 
        del self.blocks
        self.blocks = nn.Sequential(*[
            block_fn(
                anchors=anchors,
                sparse=sparse,
                dim=kwargs['embed_dim'],
                num_heads=kwargs['num_heads'],
                mlp_ratio=kwargs['mlp_ratio'],
                drop_path=dpr[i],
                norm_layer=kwargs['norm_layer'],
            )
            for i in range(kwargs['depth'])])
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2


@register_model
def anchor_deit_tiny_distilled_patch16_224(pretrained=False, anchors=30, sparse=5, **kwargs):
    model = DistilledAnchorViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_fn = AnchorBlock, anchors=anchors, sparse=sparse, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def anchor_deit_small_distilled_patch16_224(pretrained=False, anchors=30, sparse=5, **kwargs):
    model = DistilledAnchorViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_fn = AnchorBlock, anchors=anchors, sparse=sparse, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def anchor_deit_base_distilled_patch16_224(pretrained=False, anchors=30, sparse=5, **kwargs):
    model = DistilledAnchorViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_fn = AnchorBlock, anchors=anchors, sparse=sparse, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def anchor_deit_base_distilled_patch16_384(pretrained=False, anchors=30, sparse=5, **kwargs):
    model = DistilledAnchorViT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_fn = AnchorBlock, anchors=anchors, sparse=sparse, **kwargs)
    model.default_cfg = _cfg()
    return model




if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    v = anchor_deit_tiny_distilled_patch16_224(pretrained=False,
                        num_classes=1000,
                        drop_rate=0.1,
                        drop_path_rate=0.1,
                        img_size=224,
                        anchors=30,
                        sparse=5
                        )
    preds = v(img)  # (1, 1000)