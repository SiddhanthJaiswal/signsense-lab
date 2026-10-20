import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path(os.environ.get("ROBOSTAR_MODEL_DIR", ROOT / "models" / "robostar"))
BODY_MODEL = Path(os.environ.get("SMPLX_MODEL_PATH", ROOT / "models" / "smplx" / "SMPLX_NEUTRAL.npz"))
RUNTIME = ROOT / "runtime"
CACHE = RUNTIME / "cache"
