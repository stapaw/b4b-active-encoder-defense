from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class TrainingHyperParameters:
    # Training params
    train_epochs_reference_classifier: int
    train_epochs_mapper: int
    train_mb_size: int
    eval_mb_size: int


    # Optimizer
    lr_reference_classifier: float
    gamma: float
    lr_mapper: float


@dataclass
class EmbeddingTransform:
    name: str
    scale: Optional[bool]
    translation: Optional[bool]
    base_dim: Optional[int]
    pad_dim: Optional[int]
    binary_relative_dim: Optional[float]
    debinarize: Optional[bool]

@dataclass
class AffineTransform(EmbeddingTransform):
    scale: bool
    translation: bool


@dataclass
class ShufflePadTransform(EmbeddingTransform):
    base_dim: int
    pad_dim: int


@dataclass
class AffineShufflePadTransform(EmbeddingTransform):
    scale: bool
    translation: bool
    base_dim: int
    pad_dim: int


@dataclass
class BinaryTransform(EmbeddingTransform):
    base_dim: int
    binary_relative_dim: float
    debinarize: bool


@dataclass
class Embeddings:
    """
    Precalculated embedding from big encoder like DINO
    """
    train_path: str
    train_path2: Optional[str]
    test_path: str
    class_number: int
    transform: EmbeddingTransform


@dataclass
class Benchmark:
    """
    Benchmark to use.
    """

    name: str
    embeddings: Embeddings
    # model: Model
    hparams: TrainingHyperParameters
    subsets: List[int]


@dataclass
class Wandb:
    """
    Wandb configuration
    """

    enable: bool
    entity: str
    project: str
    tags: Optional[List[str]]
    name: Optional[str]


@dataclass
class Config:
    """
    Experiment configuration
    """

    benchmark: Benchmark
    wandb: Wandb
    binary: bool
    n: int
    normalize: Optional[bool]

    seed: Optional[int]
    device: str
    device_id: int
    output_dir: Optional[str]
    save_model: bool