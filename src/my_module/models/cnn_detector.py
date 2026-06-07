from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch


def load_cnn_detector(
    filepath: str | Path,
    device: str | torch.device = "cpu",
) -> Any:
    """Load a trained tremor detector model.

    Parameters
    ----------
    filepath : str | Path
        Path to a saved Keras model file.
    device : str | torch.device
        Device where the model should run.

    Returns
    -------
    Any
        Keras model configured for inference with the torch backend.
    """
    filepath = Path(filepath)
    model = keras.saving.load_model(filepath)

    model.to(device)
    model.eval()

    return model
