"""1-Weisfeiler-Lehman distinguishability study for expression graphs."""

import os

# Force a headless matplotlib backend before any submodule imports pyplot, so the
# study runs on servers/CI without a display.
os.environ.setdefault("MPLBACKEND", "Agg")
