"""Model and inference helpers copied from the project's original ViTPose_COCO notebook.

This module defines the matching architecture and helpers only. It does not train,
load checkpoints, read datasets, or perform inference when imported.
"""
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_SIZE  = 16    # 16×16 pixel patches
IMG_H       = 256   # input height
IMG_W       = 192   # input width
HM_H        = 64    # heatmap height  (IMG_H / 4)
HM_W        = 48    # heatmap width   (IMG_W / 4)
NUM_JOINTS  = 17    # COCO-17 keypoints
D_MODEL     = 768   # ViT-B hidden dim
NUM_HEADS   = 12    # attention heads
NUM_LAYERS  = 12    # transformer depth
MLP_RATIO   = 4     # FFN expansion (hidden = 768 × 4 = 3072)
DECODE_CH   = 256   # decoder intermediate channels
SIGMA       = 2.0   # Gaussian sigma for heatmap GT (heatmap pixels)

FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
JOINT_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# Affine transform helpers  ← FIXED (was passing src[:2] = 2 rows,
#                             OpenCV needs all 3 rows)
# ─────────────────────────────────────────────────────────────

def _get_3rd_point(a, b):
    """
    Given two 2-D points a and b, return a third point c such that
    (a, b, c) forms a non-degenerate basis for computing the affine transform.
    c is perpendicular to the a→b segment.
    """
    d = a - b
    return b + np.array([-d[1], d[0]], dtype=np.float32)


def _rotate_dir(vec, rot_rad):
    """Rotate a 2-D vector by rot_rad (CCW)."""
    cs, sn = math.cos(rot_rad), math.sin(rot_rad)
    return np.array([vec[0]*cs - vec[1]*sn,
                     vec[0]*sn + vec[1]*cs], dtype=np.float32)


def get_affine_transform(center, scale, rot_deg, out_size):
    """
    Compute a 2×3 affine matrix that maps a (possibly rotated)
    rectangular crop region in the original image to a fixed
    output canvas of size (out_W, out_H).

    center  : np.array([cx, cy]) — center of the crop in orig image
    scale   : float              — width of the crop region (px)
    rot_deg : float              — rotation angle in degrees
    out_size: (W, H)             — output canvas dimensions

    Returns: (2, 3) float32 affine matrix for cv2.warpAffine
    """
    dst_w, dst_h = float(out_size[0]), float(out_size[1])
    rot_rad  = math.radians(rot_deg)

    # Height of source crop region (maintain output aspect ratio)
    src_h = scale * dst_h / dst_w

    # Direction from center toward the TOP of the source region (rotated)
    src_dir = _rotate_dir(np.array([0.0, -src_h * 0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0.0, -dst_h * 0.5], dtype=np.float32)

    # ── 3 control points ──────────────────────────────────────
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    # Point 0 — crop center
    src[0] = center
    dst[0] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)

    # Point 1 — above center (in source direction, which may be rotated)
    src[1] = center + src_dir
    dst[1] = dst[0]  + dst_dir

    # Point 2 — perpendicular to 0→1  (makes the 3 points non-collinear)
    src[2] = _get_3rd_point(src[0], src[1])
    dst[2] = _get_3rd_point(dst[0], dst[1])

    # ── FIXED: pass full (3,2) arrays, NOT src[:2] ────────────
    trans = cv2.getAffineTransform(src, dst)   # src shape must be (3,2)
    return trans

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project each
    patch into a d_model-dimensional vector.

    For 256×192 input, patch_size=16:
      Patches per row  : 192 / 16 = 12
      Patches per col  : 256 / 16 = 16
      Total tokens (N) : 12 × 16  = 192

    Implemented as Conv2d(stride=patch_size), equivalent to
    the linear projection in the ViT paper.
    """
    def __init__(self, img_h, img_w, patch_size, d_model):
        super().__init__()
        self.num_h = img_h // patch_size   # 16
        self.num_w = img_w // patch_size   # 12
        self.N     = self.num_h * self.num_w  # 192

        # Linear projection implemented as stride-p Conv2d
        self.proj = nn.Conv2d(3, d_model,
                               kernel_size=patch_size,
                               stride=patch_size, bias=True)

        # Learnable positional embedding: one vector per patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N, d_model))
        self._init_pos_sincos(d_model)

        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def _init_pos_sincos(self, d_model):
        """2D sin-cos positional encoding (Dosovitskiy et al., 2021)."""
        assert d_model % 4 == 0
        gh = np.arange(self.num_h, dtype=np.float32)
        gw = np.arange(self.num_w, dtype=np.float32)
        gh, gw = np.meshgrid(gh, gw, indexing='ij')
        omega  = 1.0 / (10000 ** (np.arange(d_model//4, dtype=np.float32) / (d_model//4)))
        h_emb  = np.outer(gh.flatten(), omega)  # (N, d/4)
        w_emb  = np.outer(gw.flatten(), omega)
        pos    = np.concatenate([np.sin(h_emb), np.cos(h_emb),
                                  np.sin(w_emb), np.cos(w_emb)], axis=1)  # (N, d)
        with torch.no_grad():
            self.pos_embed.data.copy_(torch.from_numpy(pos).unsqueeze(0).float())

    def forward(self, x):
        """x: (B,3,H,W) → (B,N,d_model)"""
        x = self.proj(x)                         # (B, d, num_h, num_w)
        x = x.flatten(2).transpose(1, 2)          # (B, N, d)
        return x + self.pos_embed


# ─────────────────────────────────────────────────────────────
# B. MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────────────────────
class MHSA(nn.Module):
    """
    Multi-Head Self-Attention.

    Paper: MHSA(F) = Concat(head_1,...,head_h) W^O
           head_i  = Attention(FW^Q_i, FW^K_i, FW^V_i)
           Attention(Q,K,V) = softmax(QK^T / √d_k) V

    Uses fused QKV projection for efficiency.
    For ViT-B: d_model=768, num_heads=12 → head_dim=64.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h    = num_heads
        self.d_k  = d_model // num_heads   # 64
        self.scale= self.d_k ** -0.5       # 1/√64

        self.qkv      = nn.Linear(d_model, d_model * 3, bias=True)
        self.out_proj = nn.Linear(d_model, d_model,     bias=True)

        nn.init.trunc_normal_(self.qkv.weight,      std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight,  std=0.02)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """x: (B,N,d) → (B,N,d)"""
        B, N, C = x.shape

        # QKV: (B, N, 3, h, d_k) → 3 × (B, h, N, d_k)
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each (B, h, N, d_k)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, N, N)
        attn = F.softmax(attn, dim=-1)

        # Aggregate values: (B, h, N, d_k) → (B, N, d)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


# ─────────────────────────────────────────────────────────────
# C. FEED-FORWARD NETWORK
# ─────────────────────────────────────────────────────────────
class FFN(nn.Module):
    """
    Two-layer MLP with GELU activation.

    Paper: FFN(x) = W₂ · GELU(W₁x + b₁) + b₂

    Hidden dim = d_model × mlp_ratio = 768 × 4 = 3072  (ViT-B)
    """
    def __init__(self, d_model, mlp_ratio=4):
        super().__init__()
        h = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, h, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, d_model, bias=True)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ─────────────────────────────────────────────────────────────
# D. TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    One ViT Transformer block (Pre-LN variant).

    Paper equations:
      F'_{i+1} = F_i  + MHSA( LN(F_i)  )    ← attention sub-layer
      F_{i+1}  = F'_{i+1} + FFN( LN(F'_{i+1}) )  ← FFN sub-layer

    LayerNorm is applied BEFORE each sub-layer (Pre-LN),
    which is standard in modern ViT implementations.
    """
    def __init__(self, d_model, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn  = MHSA(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp   = FFN(d_model, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # attention + residual
        x = x + self.mlp (self.norm2(x))   # FFN       + residual
        return x


# ─────────────────────────────────────────────────────────────
# E. VITPOSE ENCODER  (Backbone)
# ─────────────────────────────────────────────────────────────
class ViTPoseEncoder(nn.Module):
    """
    ViTPose backbone = Patch Embedding + L Transformer Blocks.

    Outputs a 2D spatial feature map (B, d_model, num_h, num_w)
    ready for the pose decoder.
    """
    def __init__(self, img_h, img_w, patch_size, d_model,
                  num_heads, num_layers, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_h, img_w, patch_size, d_model)
        self.num_h = self.patch_embed.num_h
        self.num_w = self.patch_embed.num_w

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        """x: (B,3,H,W) → (B, d_model, num_h, num_w)"""
        x = self.patch_embed(x)              # (B, N, d)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                     # (B, N, d)
        B, N, C = x.shape
        return x.transpose(1,2).reshape(B, C, self.num_h, self.num_w)


# ─────────────────────────────────────────────────────────────
# F. CLASSIC POSE DECODER
# ─────────────────────────────────────────────────────────────
class ClassicDecoder(nn.Module):
    """
    Lightweight decoder (from ViTPose paper):
      TransposedConv 768→256 (×2 up) + BN + ReLU
      TransposedConv 256→256 (×2 up) + BN + ReLU
      Conv 1×1 256→17  →  17 keypoint heatmaps

    Input : (B, d_model, 16, 12)
    Output: (B, 17,      64, 48)
    """
    def __init__(self, d_model, decode_ch, num_joints):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(d_model,   decode_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(decode_ch), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(decode_ch, decode_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(decode_ch), nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(decode_ch, num_joints, 1)

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B,768,16,12) → (B,17,64,48)"""
        return self.head(self.up2(self.up1(x)))


# ─────────────────────────────────────────────────────────────
# G. FULL VITPOSE MODEL
# ─────────────────────────────────────────────────────────────
class ViTPose(nn.Module):
    """ViTPose = ViT Encoder + Classic Pose Decoder."""
    def __init__(self):
        super().__init__()
        self.encoder = ViTPoseEncoder(
            IMG_H, IMG_W, PATCH_SIZE, D_MODEL,
            NUM_HEADS, NUM_LAYERS, MLP_RATIO
        )
        self.decoder = ClassicDecoder(D_MODEL, DECODE_CH, NUM_JOINTS)

    def forward(self, x):
        """x: (B,3,256,192) → (B,17,64,48)"""
        return self.decoder(self.encoder(x))


# def decode_heatmaps(hms, center, scale):
#     """
#     Convert heatmaps (17, HM_H, HM_W) to keypoints
#     (17, 3) in original-image coordinates.
#     """
#     kps = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

#     trans = get_affine_transform(
#         center, scale, 0.0, (IMG_W, IMG_H)
#     )
#     inv_trans = cv2.invertAffineTransform(trans)

#     for j in range(NUM_JOINTS):
#         hm = hms[j]
#         flat = int(np.argmax(hm))
#         iy, ix = divmod(flat, HM_W)

#         score = float(hm[iy, ix])

#         # Sub-pixel refinement, safely handling boundaries
#         dx = 0.0
#         dy = 0.0

#         if 0 < ix < HM_W - 1:
#             dx = 0.25 * np.sign(
#                 float(hm[iy, ix + 1]) - float(hm[iy, ix - 1])
#             )

#         if 0 < iy < HM_H - 1:
#             dy = 0.25 * np.sign(
#                 float(hm[iy + 1, ix]) - float(hm[iy - 1, ix])
#             )

#         # Heatmap coordinates -> crop coordinates
#         px_crop = (ix + dx + 0.5) * (IMG_W / HM_W)
#         py_crop = (iy + dy + 0.5) * (IMG_H / HM_H)

#         # Crop coordinates -> original-image coordinates
#         pt = inv_trans @ np.array(
#             [px_crop, py_crop, 1.0], dtype=np.float32
#         )

#         kps[j] = [pt[0], pt[1], score]

#     return kps
def decode_heatmaps(hms, center, scale):
    """
    Decode heatmaps into original-image coordinates.
    hms shape: (17, 64, 48)
    Returns: (17, 3) -> x, y, confidence
    """
    kps = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

    trans = get_affine_transform(
        center, scale, 0.0, (IMG_W, IMG_H)
    )
    inv_trans = cv2.invertAffineTransform(trans)

    for j in range(NUM_JOINTS):
        hm = hms[j]

        # Find the strongest predicted location
        iy, ix = np.unravel_index(np.argmax(hm), hm.shape)
        score = float(hm[iy, ix])

        # Convert heatmap coordinates to input-crop coordinates
        px = ix * (IMG_W - 1) / (HM_W - 1)
        py = iy * (IMG_H - 1) / (HM_H - 1)

        # Convert crop coordinates to original-image coordinates
        pt = inv_trans @ np.array([px, py, 1.0])

        kps[j] = [pt[0], pt[1], score]

    return kps

BONE_COLORS = [
    (255,0,0),(255,128,0),(255,255,0),(128,255,0),(0,255,0),
    (0,255,128),(0,255,255),(0,128,255),(0,0,255),(128,0,255),
    (255,0,255),(255,0,128),(200,100,50),(50,200,100),(100,50,200),(150,150,0)
]

def draw_skeleton_on(img_rgb, kps, thresh=0.1):
    out = img_rgb.copy()
    for idx, (a, b) in enumerate(SKELETON):
        if kps[a,2] > thresh and kps[b,2] > thresh:
            cv2.line(out,
                      (int(kps[a,0]), int(kps[a,1])),
                      (int(kps[b,0]), int(kps[b,1])),
                      BONE_COLORS[idx % len(BONE_COLORS)], 2, cv2.LINE_AA)
    for j in range(NUM_JOINTS):
        if kps[j,2] > thresh:
            cv2.circle(out, (int(kps[j,0]),int(kps[j,1])), 4, (255,255,255), -1)
            cv2.circle(out, (int(kps[j,0]),int(kps[j,1])), 4, (0,0,220),     1)
    return out
