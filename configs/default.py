import dataclasses


@dataclasses.dataclass
class Config:
    # training
    batch_size: int = 8
    num_epochs: int = 1
    n_iterations: int = 10
    n_freq_train: int = 2
    learning_rate: float = 1e-3
    seed: int = 42
    dataset: str = "cifar10"  # options: "cifar10", "random"

    # image/model
    crop_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6

    # predictor
    pred_depth: int = 6
    pred_emb_dim: int = 384


def get_config():
    return Config()
