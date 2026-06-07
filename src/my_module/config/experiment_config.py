import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Experiment data settings.

    Attributes
    ----------
    dataset_dir : str
        Dataset directory path.
    train_name, val_name : str
        Training and validation dataset names.
    input_components : list[str]
        Waveform components used by the model.
    normalization_type : str
        Waveform normalization mode.
    scale_km : float
        Hypocenter output scale in kilometers.
    fs : float
        Sampling frequency.
    win_sec : float
        Window length in seconds.
    jitter_sec : float
        Jitter range in seconds.
    arrival_time : bool
        Whether arrival-time prediction is enabled.
    """

    dataset_dir: str
    train_name: str
    val_name: str
    input_components: List[str]
    normalization_type: str
    scale_km: float
    fs: float
    win_sec: float
    jitter_sec: float
    arrival_time: bool


@dataclass
class ModelConfig:
    """Vision-transformer model settings."""

    input_length: int
    patch_size: int
    stride: int
    position_emb_type: str
    emb_dim: int
    depth: int
    num_heads: int
    dropout_rate: float
    feedforward_dim: int
    hypo_num_output: int
    arrival_num_output: int
    mlp_type: str


@dataclass
class TrainConfig:
    """Training settings."""

    batch_size: int
    num_epochs: int
    learning_rate: float
    early_stopping_patience: int
    loss_function: str
    lambda_arrival: float
    seed: int
    device: str


@dataclass
class OutputConfig:
    """Experiment output settings."""

    save_dir: str
    save_best_only: bool


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Attributes
    ----------
    experiment_name : str
        Experiment identifier.
    data : DataConfig
        Data-related settings.
    model : ModelConfig
        Model-related settings.
    train : TrainConfig
        Training-related settings.
    output : OutputConfig
        Output-related settings.
    """

    experiment_name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output: OutputConfig

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        """Load an experiment configuration from JSON.

        Parameters
        ----------
        path : str or Path
            Path to the JSON configuration file.

        Returns
        -------
        ExperimentConfig
            Parsed configuration.
        """
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        data = DataConfig(**raw["data"])
        model = ModelConfig(**raw["model"])

        train_raw = raw["train"].copy()
        train = TrainConfig(**train_raw)

        output = OutputConfig(**raw["output"])

        return cls(
            experiment_name=raw["experiment_name"],
            data=data,
            model=model,
            train=train,
            output=output,
        )
