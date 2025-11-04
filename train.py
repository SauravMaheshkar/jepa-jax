import os

import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax import nnx

from configs import default
from input_pipeline import create_input_fns
from model import IJEPA


def smooth_l1_loss(
    pred: jnp.ndarray, target: jnp.ndarray, beta: float = 1.0
) -> jnp.ndarray:
    diff = jnp.abs(pred - target)
    loss = jnp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return jnp.mean(loss)


def sample_masks(rng, B: int, num_patches: int, ratio_keep: float = 0.5):
    k = int(num_patches * ratio_keep)
    m = jnp.arange(num_patches)
    rngs = jax.random.split(rng, B)
    enc_masks, pred_masks = [], []
    for r in rngs:
        perm = jax.random.permutation(r, m)
        enc_masks.append(perm[:k])
        pred_masks.append(perm[k:])
    enc_masks = jnp.stack(enc_masks, axis=0)  # [B, K]
    pred_masks = jnp.stack(pred_masks, axis=0)  # [B, N-K]
    return [enc_masks], [pred_masks]


@nnx.jit
def train_step(
    model: IJEPA,
    optimizer: nnx.ModelAndOptimizer,
    imgs: jax.Array,
    masks_enc,
    masks_pred,
):
    def loss_fn(m):
        rngs = nnx.Rngs(0)
        h = m.forward_target(imgs, masks_pred, rngs=rngs)
        z = m.forward_context(imgs, masks_enc, masks_pred, rngs=rngs)
        return smooth_l1_loss(z, h)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(
    model: IJEPA,
    imgs: jax.Array,
    masks_enc,
    masks_pred,
):
    def loss_fn(m):
        rngs = nnx.Rngs(0)
        h = m.forward_target(imgs, masks_pred, rngs=rngs)
        z = m.forward_context(imgs, masks_enc, masks_pred, rngs=rngs)
        return smooth_l1_loss(z, h)

    loss = loss_fn(model)
    return loss


def train_and_evaluate(config: default.Config, workdir: str) -> None:
    workdir = os.path.abspath(workdir)

    img_size = config.crop_size
    patch_size = config.patch_size
    embed_dim = config.embed_dim
    depth = config.depth
    num_heads = config.num_heads
    pred_depth = config.pred_depth
    pred_emb_dim = config.pred_emb_dim

    rng = jax.random.PRNGKey(config.seed)
    rng, rp, rd = jax.random.split(rng, 3)
    rngs = nnx.Rngs(params=rp, dropout=rd, drop_path=rd)

    model = IJEPA(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        predictor_embed_dim=pred_emb_dim,
        predictor_depth=pred_depth,
        rngs=rngs,
    )

    logging.info(
        f"Total parameters: {sum(p.size for p in jax.tree.leaves(nnx.state(model))):_}"
    )

    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(config.learning_rate))

    steps = config.num_epochs * config.n_iterations
    num_patches = (img_size // patch_size) ** 2

    next_train_batch, next_eval_batch = create_input_fns(config)

    for step in range(steps):
        rng, rimg, rmask, reval = jax.random.split(rng, 4)
        imgs = jnp.asarray(next_train_batch(rimg))
        masks_enc, masks_pred = sample_masks(
            rmask, config.batch_size, num_patches, ratio_keep=0.5
        )
        loss = train_step(model, optimizer, imgs, masks_enc, masks_pred)
        if (step + 1) % config.n_freq_train == 0:
            # Eval on one batch
            eval_imgs = jnp.asarray(next_eval_batch(reval))
            ev_masks_enc, ev_masks_pred = sample_masks(
                reval, config.batch_size, num_patches, ratio_keep=0.5
            )
            eval_loss = eval_step(model, eval_imgs, ev_masks_enc, ev_masks_pred)
            print(
                f"Step {step + 1} | Loss: {float(loss):.4f} | Eval Loss: {float(eval_loss):.4f}"  # noqa: E501
            )

    # Save params
    import orbax.checkpoint as ocp

    os.makedirs(workdir, exist_ok=True)
    state = nnx.state(model)
    ocp.PyTreeCheckpointer().save(f"{workdir}/ijepa", state)
