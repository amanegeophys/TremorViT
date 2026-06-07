import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConfig:
    """Project-level base settings."""

    project_name: str
    base_catalog: str


@dataclass
class WaveformConfig:
    """Waveform processing settings."""

    freqmin_for_detector: float
    freqmax_for_detector: float
    sampling_rate: int
    duration_sec: float


@dataclass
class SpectrogramConfig:
    """Spectrogram generation settings."""

    fft_window_sec: float
    overlap_rate: float
    freqmin: int
    freqmax: int
    normalize_type: str


@dataclass
class SacConfig:
    """SAC file location settings."""

    year_to_path: dict[int, str]
    component_channels: dict[str, str]


@dataclass
class CatalogConfig:
    """Catalog construction and split settings."""

    dataset_dir: str
    win_sec: float
    prob_threshold: float
    max_distance_km: float
    min_val_count: int
    min_test_count: int
    min_total_count: int
    start_year: int
    end_year: int
    train: list | float
    val: list | float
    test: list | float
    jitter_sec: int


@dataclass
class ProjectConfig:
    """Complete project configuration.

    Attributes
    ----------
    base : BaseConfig
        Base project settings.
    waveform : WaveformConfig
        Waveform processing settings.
    spectrogram : SpectrogramConfig
        Spectrogram generation settings.
    catalog : CatalogConfig
        Catalog construction settings.
    """

    base: BaseConfig
    waveform: WaveformConfig
    spectrogram: SpectrogramConfig
    sac: SacConfig
    catalog: CatalogConfig

    @classmethod
    def from_file(cls, path: str | Path) -> "ProjectConfig":
        """Load a project configuration from JSON.

        Parameters
        ----------
        path : str or Path
            Path to the JSON configuration file.

        Returns
        -------
        ProjectConfig
            Parsed configuration.
        """
        path = Path(path).expanduser().resolve()

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        bs = BaseConfig(**raw["base"])
        wf = WaveformConfig(**raw["waveform"])
        sg = SpectrogramConfig(**raw["spectrogram"])
        sac_raw = raw["sac"]
        sac = SacConfig(
            year_to_path={
                int(year): str(Path(path).expanduser())
                for year, path in sac_raw["year_to_path"].items()
            },
            component_channels=dict(sac_raw["component_channels"]),
        )
        cg = CatalogConfig(**raw["catalog"])

        return cls(base=bs, waveform=wf, spectrogram=sg, sac=sac, catalog=cg)
