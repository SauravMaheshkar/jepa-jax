import dataclasses


@dataclasses.dataclass
class Config:
    # training
    batch_size: int = 4
    num_epochs: int = 1
    n_iterations: int = 50
    n_freq_train: int = 10
    learning_rate: float = 1e-3
    seed: int = 42
    dataset: str = "random"  # options: "cifar10", "random"

    # image/model
    crop_size: int = 128
    patch_size: int = 16
    embed_dim: int = 192
    depth: int = 3
    num_heads: int = 3

    # predictor
    pred_depth: int = 3
    pred_emb_dim: int = 192


def get_config():
    return Config()
