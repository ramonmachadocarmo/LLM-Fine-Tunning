try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from src.config import get_config
from src.training import Trainer


def main() -> None:
    config = get_config()
    Trainer(config).train()


if __name__ == "__main__":
    main()
