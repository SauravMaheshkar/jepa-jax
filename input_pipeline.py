from typing import Callable, Tuple

import grain
import jax
import jax.numpy as jnp
import numpy as np

from configs import default as default_config


ArrayBCHW = jax.Array


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover - import error surfaced at runtime
        raise RuntimeError(
            "Pillow is required for image resizing. Please add 'pillow' to dependencies."
        ) from e

    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil)


def _create_random_iters(
    cfg: default_config.Config,
) -> Tuple[Callable[[], np.ndarray], Callable[[], np.ndarray]]:
    img_size = cfg.crop_size
    batch_size = cfg.batch_size

    def next_train_batch(rng_key: jax.Array) -> np.ndarray:
        arr = jax.random.normal(
            rng_key, (batch_size, img_size, img_size, 3), dtype=jnp.float32
        )
        return np.asarray(arr)

    def next_eval_batch(rng_key: jax.Array) -> np.ndarray:
        arr = jax.random.normal(
            rng_key, (batch_size, img_size, img_size, 3), dtype=jnp.float32
        )
        return np.asarray(arr)

    return next_train_batch, next_eval_batch


def _create_cifar10_iters(
    cfg: default_config.Config,
) -> Tuple[Callable[[], np.ndarray], Callable[[], np.ndarray]]:
    from datasets import load_dataset

    img_size = cfg.crop_size
    batch_size = cfg.batch_size

    def preprocess(features):
        img = np.asarray(features["image"]).astype(np.uint8)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = _resize_image(img, img_size)
        return (img.astype(np.float32) / 255.0).astype(np.float32)

    # Build TF-free random access sources using Hugging Face datasets
    ds_train_hf = load_dataset("cifar10", split="train")
    ds_eval_hf = load_dataset("cifar10", split="test")

    class HFRandomAccess(grain.sources.RandomAccessDataSource):
        def __init__(self, hf_ds):
            self._ds = hf_ds

        def __getitem__(self, idx):
            ex = self._ds[int(idx)]
            # HF provides PIL Image in ex["img"] for cifar10
            img = np.array(ex["img"]).astype(np.uint8)
            return {"image": img}

        def __len__(self):
            return len(self._ds)

    source_train = HFRandomAccess(ds_train_hf)
    source_eval = HFRandomAccess(ds_eval_hf)

    ds_train = (
        grain.MapDataset.source(source_train)
        .seed(cfg.seed)
        .shuffle()
        .map(preprocess)
        .batch(batch_size)
        .to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=500))
    )
    ds_eval = (
        grain.MapDataset.source(source_eval)
        .map(preprocess)
        .batch(batch_size)
        .to_iter_dataset(grain.ReadOptions(num_threads=8, prefetch_buffer_size=200))
    )

    train_iter = iter(ds_train)
    eval_iter = iter(ds_eval)

    def next_train_batch(_: jax.Array) -> np.ndarray:
        return next(train_iter)

    def next_eval_batch(_: jax.Array) -> np.ndarray:
        return next(eval_iter)

    return next_train_batch, next_eval_batch


def create_input_fns(
    cfg: default_config.Config,
) -> Tuple[Callable[[jax.Array], np.ndarray], Callable[[jax.Array], np.ndarray]]:
    """
    Returns two callables: next_train_batch(rng) and next_eval_batch(rng).
    They produce numpy arrays shaped (B, H, W, 3), float32 in [0, 1]
    for cifar10 or Gaussian for random.
    """
    dataset = getattr(cfg, "dataset", "cifar10").lower()
    if dataset == "random":
        return _create_random_iters(cfg)
    if dataset == "cifar10":
        return _create_cifar10_iters(cfg)
    raise ValueError(f"Unsupported dataset: {dataset}")
