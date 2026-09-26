"""Application-wide configuration.

Values are read from environment variables so they can be changed without code edits.
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Anchor for actuarial discounting: 10-year Indian Government Bond (G-Sec) yield.
GSEC_10Y_YIELD: float = float(os.environ.get("OPTIFIN_GSEC_10Y_YIELD", "0.07"))
