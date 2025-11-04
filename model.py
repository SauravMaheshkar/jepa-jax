import os
from typing import Optional
from urllib.request import urlretrieve

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def apply_masks(x, masks):
    """
    :param x: array of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of jnp arrays containing indices of patches in [N] to keep
                  Each mask is of shape [B, K] where K is the number of patches to keep
    """
    all_x = []
    for m in masks:
        # m is [B, K], need to create [B, K, D] for take_along_axis
        idx = jnp.expand_dims(m, axis=-1)  # [B, K, 1]
        idx = idx.repeat(x.shape[-1], axis=-1)  # [B, K, D]
        gathered = jnp.take_along_axis(x, idx, axis=1)
        all_x.append(gathered)
    return jnp.concatenate(all_x, axis=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    parts = []
    for i in range(N):
        part = x[i * B : (i + 1) * B]
        parts.append(jnp.tile(part, (repeat, 1, 1)))
    return jnp.concatenate(parts, axis=0)


def _load_pytorch_checkpoint(checkpoint_path: str):
    """Load a PyTorch checkpoint file."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load pretrained checkpoints. ")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        elif "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        else:
            return checkpoint
    return checkpoint


def _convert_pytorch_to_jax(pytorch_state_dict, model):
    """Convert PyTorch state dict to JAX/Flax NNX state."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to convert checkpoints. "
            "Install it with: pip install torch"
        )

    jax_state = nnx.state(model)

    def convert_param(param, key: str):
        if isinstance(param, torch.Tensor):
            arr = param.detach().cpu().numpy()
            if "proj.weight" in key or "qkv.weight" in key or "fc" in key:
                if arr.ndim == 2:
                    arr = arr.transpose()
                elif arr.ndim == 4:
                    arr = arr.transpose(2, 3, 1, 0)
            return jnp.array(arr)
        return param

    def normalize_key(key: str) -> str:
        key = key.replace("module.", "")
        return key

    normalized_state_dict = {}
    for pytorch_key, value in pytorch_state_dict.items():
        normalized_key = normalize_key(pytorch_key)
        normalized_state_dict[normalized_key] = convert_param(value, pytorch_key)

    def set_param_by_path(state, path: list, value):
        if len(path) == 1:
            key = path[0]
            if key in state:
                if isinstance(state[key], nnx.Param):
                    state[key].value = value
                else:
                    state[key] = value
        else:
            key = path[0]
            if key in state:
                if isinstance(state[key], dict):
                    set_param_by_path(state[key], path[1:], value)
                elif isinstance(state[key], (list, tuple)):
                    if path[1].isdigit():
                        idx = int(path[1])
                        if idx < len(state[key]):
                            if isinstance(state[key][idx], dict):
                                set_param_by_path(state[key][idx], path[2:], value)
            elif isinstance(state, dict):
                for state_key, state_value in state.items():
                    if isinstance(state_value, dict) and key in state_value:
                        set_param_by_path(state_value, path[1:], value)
                    elif isinstance(state_value, (list, tuple)):
                        if path[1].isdigit():
                            idx = int(path[1])
                            if idx < len(state_value) and isinstance(
                                state_value[idx], dict
                            ):
                                set_param_by_path(state_value[idx], path[2:], value)

    for key, value in normalized_state_dict.items():
        path = key.split(".")
        set_param_by_path(jax_state, path, value)

    return jax_state


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> jnp.ndarray:
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return jnp.array(emb)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class DropPath(nnx.Module):
    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob

    def __call__(self, x, *, rngs=None, deterministic: bool = False):
        if self.drop_prob == 0.0 or deterministic:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = jax.random.uniform(
            rngs["drop_path"], shape, dtype=x.dtype, minval=0.0, maxval=1.0
        )
        mask = jnp.floor(keep_prob + r)
        return (x / keep_prob) * mask


class MLP(nnx.Module):
    def __init__(
        self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, *, rngs: nnx.Rngs
    ):
        hidden = int(dim * mlp_ratio)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)
        self.drop = nnx.Dropout(rate=drop)

    def __call__(self, x, *, rngs):
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.drop(x, rngs=rngs)
        x = self.fc2(x)
        x = self.drop(x, rngs=rngs)
        return x


class Attention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.attn_drop = nnx.Dropout(rate=attn_drop)
        self.proj_drop = nnx.Dropout(rate=proj_drop)

    def __call__(self, x, *, rngs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, rngs=rngs)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, rngs=rngs)
        return x


class Block(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop, rngs=rngs)
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = MLP(dim, mlp_ratio, drop, rngs=rngs)
        self.drop_path2 = DropPath(drop_prob=drop_path)

    def __call__(self, x, *, rngs, deterministic: bool = False):
        x = x + self.drop_path1(
            self.attn(self.norm1(x), rngs=rngs), rngs=rngs, deterministic=deterministic
        )
        x = x + self.drop_path2(
            self.mlp(self.norm2(x), rngs=rngs), rngs=rngs, deterministic=deterministic
        )
        return x


class PatchEmbed(nnx.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.ps = patch_size
        self.proj = nnx.Conv(
            in_chans,
            embed_dim,
            (patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.proj(x)  # NHWC
        B, H, W, C = x.shape
        return x.reshape(B, H * W, C)


class Encoder(nnx.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim, rngs=rngs)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nnx.Param(
            get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5))[None, :, :]
        )
        self.blocks = nnx.List(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_path_rate * i / max(depth - 1, 1) if depth > 1 else 0.0,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

    def __call__(self, x, masks=None, *, rngs, deterministic: bool = False):
        x = self.patch_embed(x)
        x = x + self.pos_embed.value
        if masks is not None:
            x = apply_masks(x, masks)
        for blk in self.blocks:
            x = blk(x, rngs=rngs, deterministic=deterministic)
        x = self.norm(x)
        return x


class Predictor(nnx.Module):
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_proj = nnx.Linear(embed_dim, predictor_embed_dim, rngs=rngs)
        self.mask_token = nnx.Param(jnp.zeros((predictor_embed_dim,)))
        self.pos_embed = nnx.Param(
            get_2d_sincos_pos_embed(predictor_embed_dim, int(num_patches**0.5))[
                None, :, :
            ]
        )
        self.blocks = nnx.List(
            [
                Block(
                    predictor_embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_path_rate * i / max(depth - 1, 1) if depth > 1 else 0.0,
                    rngs=rngs,
                )
                for i in range(depth)
            ]
        )
        self.norm = nnx.LayerNorm(predictor_embed_dim, rngs=rngs)
        self.out_proj = nnx.Linear(predictor_embed_dim, embed_dim, rngs=rngs)

    def __call__(self, enc_tokens, masks_x, masks, *, rngs, deterministic: bool = False):
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks, list):
            masks = [masks]
        B = len(enc_tokens) // len(masks_x)
        x = self.in_proj(enc_tokens)
        x += apply_masks(jnp.tile(self.pos_embed.value, (B, 1, 1)), masks_x)
        _, N_ctxt, _ = x.shape
        pos_embs = apply_masks(jnp.tile(self.pos_embed.value, (B, 1, 1)), masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = jnp.tile(
            self.mask_token.value[None, None, :],
            (pos_embs.shape[0], pos_embs.shape[1], 1),
        )
        pred_tokens += pos_embs
        x = jnp.tile(x, (len(masks), 1, 1))
        x = jnp.concatenate([x, pred_tokens], axis=1)
        for blk in self.blocks:
            x = blk(x, rngs=rngs, deterministic=deterministic)
        x = self.norm(x)
        x = x[:, N_ctxt:]
        x = self.out_proj(x)
        return x


class IJEPA(nnx.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        rngs: nnx.Rngs,
    ):
        self.encoder = Encoder(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            rngs=rngs,
        )
        num_patches = (img_size // patch_size) ** 2
        self.predictor = Predictor(
            num_patches,
            embed_dim,
            predictor_embed_dim,
            predictor_depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            rngs=rngs,
        )
        self.target_encoder = Encoder(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            rngs=rngs,
        )

    def forward_target(self, imgs, masks_pred, *, rngs):
        with jax.named_scope("target"):
            h = self.target_encoder(imgs, masks=None, rngs=rngs)
            h = (h - jnp.mean(h, axis=-1, keepdims=True)) / (
                jnp.std(h, axis=-1, keepdims=True) + 1e-6
            )
            B = h.shape[0]
            h = apply_masks(h, masks_pred)
            h = repeat_interleave_batch(h, B, repeat=len(masks_pred))
            return h

    def forward_context(self, imgs, masks_enc, masks_pred, *, rngs):
        z = self.encoder(imgs, masks=masks_enc, rngs=rngs)
        z = self.predictor(z, masks_enc, masks_pred, rngs=rngs)
        return z

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: Optional[str] = None,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ):
        """Load a pretrained IJEPA model.

        Args:
            model_name: Name of the pretrained model. Available models:
                - "IN1K-vit.h.14-300e": ViT-H, patch 14x14, 224x224, ImageNet-1K
                - "IN1K-vit.h.16-448px-300e": ViT-H, patch 16x16, 448x448, ImageNet-1K
                - "IN22K-vit.h.14-900e": ViT-H, patch 14x14, 224x224, ImageNet-22K
                - "IN22K-vit.g.16-600e": ViT-g, patch 16x16, 224x224, ImageNet-22K
            cache_dir: Directory to cache downloaded models. Defaults to ~/.cache/ijepa
            rngs: Optional Rngs for model initialization. If None, uses default seed.

        Returns:
            IJEPA model with pretrained weights loaded.
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/ijepa")
        os.makedirs(cache_dir, exist_ok=True)

        model_configs = {
            "IN1K-vit.h.14-300e": {
                "url": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
                "img_size": 224,
                "patch_size": 14,
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
                "predictor_embed_dim": 512,
                "predictor_depth": 6,
            },
            "IN1K-vit.h.16-448px-300e": {
                "url": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar",
                "img_size": 448,
                "patch_size": 16,
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
                "predictor_embed_dim": 512,
                "predictor_depth": 6,
            },
            "IN22K-vit.h.14-900e": {
                "url": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar",
                "img_size": 224,
                "patch_size": 14,
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 16,
                "predictor_embed_dim": 512,
                "predictor_depth": 6,
            },
            "IN22K-vit.g.16-600e": {
                "url": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar",
                "img_size": 224,
                "patch_size": 16,
                "embed_dim": 1408,
                "depth": 40,
                "num_heads": 16,
                "predictor_embed_dim": 512,
                "predictor_depth": 6,
            },
        }

        if model_name not in model_configs:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available models: {list(model_configs.keys())}"
            )

        config = model_configs[model_name]
        checkpoint_path = os.path.join(cache_dir, f"{model_name}.pth.tar")

        if not os.path.exists(checkpoint_path):
            print(f"Downloading {model_name} checkpoint...")
            urlretrieve(config["url"], checkpoint_path)
            print(f"Downloaded to {checkpoint_path}")

        if rngs is None:
            rng = jax.random.PRNGKey(42)
            rng, rp, rd = jax.random.split(rng, 3)
            rngs = nnx.Rngs(params=rp, dropout=rd, drop_path=rd)

        model = cls(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=3,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            predictor_embed_dim=config["predictor_embed_dim"],
            predictor_depth=config["predictor_depth"],
            rngs=rngs,
        )

        state_dict = _load_pytorch_checkpoint(checkpoint_path)
        jax_state = _convert_pytorch_to_jax(state_dict, model)
        nnx.update(model, jax_state)

        return model
