import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.util import can_softmax

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def degree(B, dim):
    bacth, n_heads, N, anchor_num = B.shape
    degreeB = torch.sum(B, dim=-2).pow(-1).unsqueeze(-1)
    degreeB[torch.isinf(degreeB)] = 0.0
    degreeB = degreeB.repeat(1, 1, 1, dim)
    return degreeB

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class AnchorAttention(nn.Module):
    def __init__(self, dim, anchors=5, heads = 8, dim_head = 64, dropout = 0., k_sparse=5):
        super(AnchorAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.k_sparse = k_sparse
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.x_to_emb = nn.Linear(dim, inner_dim * 2, bias=False)
        self.dist_anchor_B = nn.Linear(dim_head, anchors, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        emv_v = self.x_to_emb(x).chunk(2, dim=-1)
        q_k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), emv_v)
        anchor_B = self.dist_anchor_B(q_k) * self.scale
        anchor_attn = anchor_B.softmax(dim=-1)
        attn_degree = degree(anchor_attn, dim=q_k.shape[-1])
        anchor_attn_T = anchor_attn.transpose(3, 2)

        out = anchor_attn_T.matmul(v)
        out = attn_degree.mul(out)
        out = anchor_attn.matmul(out)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class AnchorTransformer(nn.Module): 
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, anchors=5, dropout=0., k_sparse=5):
        super(AnchorTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, AnchorAttention(dim, anchors=anchors, heads=heads, dim_head=dim_head, dropout=dropout, k_sparse=k_sparse)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class AnchorViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, anchors=5, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., k_sparse=5):
        super(AnchorViT, self).__init__()
        image_height, image_width = pair(image_size) 
        patch_height, patch_width = pair(patch_size)  

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = AnchorTransformer(dim, depth, heads, dim_head, mlp_dim, anchors=anchors, dropout=dropout, k_sparse=k_sparse)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)  
        b, n, _ = x.shape  

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)