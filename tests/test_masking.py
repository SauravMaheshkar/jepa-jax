import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from model import IJEPA, Encoder, apply_masks


@pytest.mark.parametrize(
    "case",
    [
        {"name": "constant", "fill": 0.0},
        {"name": "constant", "fill": 1.0},
        {"name": "empty"},
        {"name": "identity"},
        {"name": "multiple"},
    ],
)
def test_apply_masks_cases(case):
    """Combined tests for apply_masks covering constants, empty, identity, multiple."""
    B, N, D = 2, 16, 8

    if case.get("name") == "constant":
        fill_value = case["fill"]
        x = jnp.full((B, N, D), fill_value, dtype=jnp.float32)
        mask1 = jnp.tile(jnp.array([[0, 5, 10]]), (B, 1))
        mask2 = jnp.tile(jnp.array([[1, 7, 12]]), (B, 1))
        masks = [mask1, mask2]
        result = apply_masks(x, masks)
        assert result.shape == (len(masks) * B, masks[0].shape[1], D)
        assert jnp.allclose(result, fill_value)
        expected_sum = len(masks) * B * masks[0].shape[1] * D * fill_value
        assert jnp.allclose(jnp.sum(result), expected_sum)

    elif case.get("name") == "empty":
        x = jnp.ones((B, N, D), dtype=jnp.float32)
        mask = jnp.empty((B, 0), dtype=jnp.int32)
        result = apply_masks(x, [mask])
        assert result.shape == (B, 0, D)

    elif case.get("name") == "identity":
        x = jnp.arange(B * N * D).reshape(B, N, D).astype(jnp.float32)
        masks = [jnp.tile(jnp.arange(N)[None, :], (B, 1))]
        result = apply_masks(x, masks)
        assert result.shape == (B, N, D)
        np.testing.assert_array_equal(result[0], x[0])

    elif case.get("name") == "multiple":
        x = jnp.arange(B * N * D).reshape(B, N, D).astype(jnp.float32)
        masks = [
            jnp.tile(jnp.array([[0, 1, 2]]), (B, 1)),
            jnp.tile(jnp.array([[5, 6, 7]]), (B, 1)),
            jnp.tile(jnp.array([[10, 11, 12]]), (B, 1)),
        ]
        result = apply_masks(x, masks)
        assert result.shape[0] == len(masks) * B
        assert result.shape[1] == masks[0].shape[1]
        np.testing.assert_array_equal(result[0, 0], x[0, 0])
        np.testing.assert_array_equal(result[B, 0], x[0, 5])
        np.testing.assert_array_equal(result[2 * B, 0], x[0, 10])


@pytest.mark.parametrize("img_value", [0.0, 1.0])
@pytest.mark.parametrize("mode", ["half", "full"])
def test_encoder_masking_modes(img_value, mode):
    """Combined tests for Encoder masking with constant images for half/full masks."""
    img_size = 32
    patch_size = 8
    embed_dim = 64
    batch_size = 2
    num_patches = (img_size // patch_size) ** 2

    rng = jax.random.PRNGKey(42)
    rng, rp, rd = jax.random.split(rng, 3)
    rngs = nnx.Rngs(params=rp, dropout=rd, drop_path=rd)

    encoder = Encoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        rngs=rngs,
    )

    imgs = jnp.full((batch_size, img_size, img_size, 3), img_value, dtype=jnp.float32)

    rng, rd = jax.random.split(rng)
    rngs = nnx.Rngs(dropout=rd, drop_path=rd)

    if mode == "full":
        output_no_mask = encoder(imgs, masks=None, rngs=rngs, deterministic=True)
        masks = [jnp.tile(jnp.arange(num_patches)[None, :], (batch_size, 1))]
        output_full_mask = encoder(imgs, masks=masks, rngs=rngs, deterministic=True)
        np.testing.assert_allclose(output_no_mask[0], output_full_mask[0], rtol=1e-5)
    else:
        masks = [jnp.tile(jnp.arange(num_patches // 2)[None, :], (batch_size, 1))]
        output = encoder(imgs, masks=masks, rngs=rngs, deterministic=True)
        assert output.shape[0] == len(masks) * batch_size
        assert output.shape[1] == masks[0].shape[1]
        assert output.shape[2] == embed_dim
        if img_value == 0.0:
            assert not jnp.allclose(output, 0.0)
        else:
            assert not jnp.allclose(output, output[0, 0])
        assert not jnp.any(jnp.isnan(output))
        assert not jnp.any(jnp.isinf(output))


@pytest.mark.parametrize("img_value", [0.0, 1.0])
def test_ijepa_forward_constant_images(img_value):
    """Test IJEPA forward pass with constant-value images."""
    img_size = 32
    patch_size = 8
    embed_dim = 64
    batch_size = 2
    num_patches = (img_size // patch_size) ** 2

    rng = jax.random.PRNGKey(42)
    rng, rp, rd = jax.random.split(rng, 3)
    rngs = nnx.Rngs(params=rp, dropout=rd, drop_path=rd)

    model = IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=2,
        num_heads=4,
        predictor_embed_dim=64,
        predictor_depth=2,
        qkv_bias=True,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        rngs=rngs,
    )

    imgs = jnp.full((batch_size, img_size, img_size, 3), img_value, dtype=jnp.float32)
    masks_enc = [jnp.tile(jnp.arange(num_patches // 2)[None, :], (batch_size, 1))]
    masks_pred = [
        jnp.tile(jnp.arange(num_patches // 2, num_patches)[None, :], (batch_size, 1))
    ]

    rng, rd = jax.random.split(rng)
    rngs = nnx.Rngs(dropout=rd, drop_path=rd)

    h = model.forward_target(imgs, masks_pred, rngs=rngs)
    z = model.forward_context(imgs, masks_enc, masks_pred, rngs=rngs)

    assert h.shape == z.shape, "Target and context outputs should have same shape"
    assert h.shape[1] == masks_pred[0].shape[1], (
        "Output patches should match prediction mask"
    )

    # Check for NaN/Inf
    assert not jnp.any(jnp.isnan(h)), "No NaN in target output"
    assert not jnp.any(jnp.isinf(h)), "No Inf in target output"
    assert not jnp.any(jnp.isnan(z)), "No NaN in context output"
    assert not jnp.any(jnp.isinf(z)), "No Inf in context output"

    # Check numerical stability
    assert jnp.abs(h).max() < 1e6, "Target output should be numerically stable"
    assert jnp.abs(z).max() < 1e6, "Context output should be numerically stable"
