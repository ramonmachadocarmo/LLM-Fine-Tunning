import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.hf_auth import load_dotenv_into_environ  # noqa: E402

load_dotenv_into_environ()
