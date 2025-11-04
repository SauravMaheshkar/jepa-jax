import os

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from model import IJEPA


def sample_masks(rng, B: int, num_patches: int, ratio_keep: float = 0.5):
    k = int(num_patches * ratio_keep)
    m = jnp.arange(num_patches)
    rngs = jax.random.split(rng, B)
    enc_masks, pred_masks = [], []
    for r in rngs:
        perm = jax.random.permutation(r, m)
        enc_masks.append(perm[:k])
        pred_masks.append(perm[k:])
    enc_masks = jnp.stack(enc_masks, axis=0)
    pred_masks = jnp.stack(pred_masks, axis=0)
    return [enc_masks], [pred_masks]


PRETRAINED_MODELS = [
    "IN1K-vit.h.14-300e",
    # enable these for the full test suite
    # disabled to avoid OOM on GH action runners
    # "IN1K-vit.h.16-448px-300e",
    # "IN22K-vit.h.14-900e",
    # "IN22K-vit.g.16-600e",
]


@pytest.mark.large
@pytest.mark.parametrize("model_name", PRETRAINED_MODELS)
def test_pretrained_model_loading(model_name, tmp_path):
    """Test that pretrained models can be loaded and weights are valid."""
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model = IJEPA.from_pretrained(model_name, cache_dir=cache_dir)

    state = nnx.state(model)

    def get_param_size(param):
        if hasattr(param, "value"):
            param = param.value
        if isinstance(param, jnp.ndarray):
            return param.size
        return 0

    total_params = sum(get_param_size(p) for p in jax.tree.leaves(state))

    assert total_params > 0, f"Model {model_name} should have parameters"

    def check_param(param):
        if hasattr(param, "value"):
            param = param.value
        if isinstance(param, jnp.ndarray):
            assert not jnp.all(jnp.isnan(param)), (
                f"NaN found in parameters for {model_name}"
            )
            assert not jnp.all(jnp.isinf(param)), (
                f"Inf found in parameters for {model_name}"
            )
            assert param.size > 0, f"Empty parameter found for {model_name}"

    jax.tree.map(check_param, state)


@pytest.mark.large
@pytest.mark.parametrize("model_name", PRETRAINED_MODELS)
def test_pretrained_forward_pass(model_name, tmp_path):
    """Test forward pass on pretrained models."""
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model = IJEPA.from_pretrained(model_name, cache_dir=cache_dir)

    model_configs = {
        "IN1K-vit.h.14-300e": {"img_size": 224, "patch_size": 14},
        "IN1K-vit.h.16-448px-300e": {"img_size": 448, "patch_size": 16},
        "IN22K-vit.h.14-900e": {"img_size": 224, "patch_size": 14},
        "IN22K-vit.g.16-600e": {"img_size": 224, "patch_size": 16},
    }

    config = model_configs[model_name]
    img_size = config["img_size"]
    patch_size = config["patch_size"]
    batch_size = 2
    num_patches = (img_size // patch_size) ** 2

    rng = jax.random.PRNGKey(42)
    rng, rimg, rmask = jax.random.split(rng, 3)

    imgs = jax.random.normal(rimg, (batch_size, img_size, img_size, 3), dtype=jnp.float32)

    masks_enc, masks_pred = sample_masks(rmask, batch_size, num_patches, ratio_keep=0.5)

    rng, rp, rd = jax.random.split(rng, 3)
    rngs = nnx.Rngs(params=rp, dropout=rd, drop_path=rd)

    h = model.forward_target(imgs, masks_pred, rngs=rngs)
    z = model.forward_context(imgs, masks_enc, masks_pred, rngs=rngs)

    assert h.shape[0] == z.shape[0], "Output shapes should match"
    assert h.shape[1] == z.shape[1], "Output shapes should match"
    assert h.shape[2] == z.shape[2], "Output shapes should match"

    assert not jnp.any(jnp.isnan(h)), f"NaN in target output for {model_name}"
    assert not jnp.any(jnp.isinf(h)), f"Inf in target output for {model_name}"
    assert not jnp.any(jnp.isnan(z)), f"NaN in context output for {model_name}"
    assert not jnp.any(jnp.isinf(z)), f"Inf in context output for {model_name}"

    assert jnp.abs(h).max() < 1e6, f"Target output too large for {model_name}"
    assert jnp.abs(z).max() < 1e6, f"Context output too large for {model_name}"


@pytest.mark.large
@pytest.mark.parametrize("model_name", PRETRAINED_MODELS)
def test_pretrained_weight_statistics(model_name, tmp_path):
    """Test that loaded weights have reasonable statistics."""
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model = IJEPA.from_pretrained(model_name, cache_dir=cache_dir)
    state = nnx.state(model)

    param_stats = []

    def collect_stats(param):
        if hasattr(param, "value"):
            param = param.value
        if isinstance(param, jnp.ndarray) and param.size > 0:
            param_mean = float(jnp.mean(jnp.abs(param)))
            param_std = float(jnp.std(param))
            param_stats.append((param_mean, param_std))

    jax.tree.map(collect_stats, state)

    assert len(param_stats) > 0, f"Should have parameters for {model_name}"

    all_zeros = all(mean < 1e-10 for mean, _ in param_stats)
    assert not all_zeros, f"All parameters are near zero for {model_name}"

    reasonable_std = any(0.01 < std < 100 for _, std in param_stats)
    assert reasonable_std, f"Parameters should have reasonable std dev for {model_name}"
