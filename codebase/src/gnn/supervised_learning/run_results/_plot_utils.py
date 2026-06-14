from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(output_path: Path, *, bbox_inches: str = "tight") -> None:
    """Save the current figure as PNG and as SVG/PDF inside a 'vector/' subdirectory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches=bbox_inches)

    vector_dir = output_path.parent / "vector"
    vector_dir.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem
    plt.savefig(vector_dir / f"{stem}.svg", bbox_inches=bbox_inches)
    plt.savefig(vector_dir / f"{stem}.pdf", bbox_inches=bbox_inches)
