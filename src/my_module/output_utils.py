from collections.abc import Iterable
from pathlib import Path


def check_output_paths(paths: Iterable[Path], *, overwrite: bool = False) -> None:
    """Reject existing outputs unless replacement was explicitly requested."""
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        names = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            f"Output files already exist:\n{names}\n"
            "Choose a new output directory or pass --overwrite to replace them."
        )
