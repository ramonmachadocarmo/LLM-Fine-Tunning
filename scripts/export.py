import argparse

try:
    from . import _bootstrap  # noqa: F401
except ImportError:
    import _bootstrap  # noqa: F401

from src.config.loader import load_yaml
from src.export import run_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and Convert Fine-Tuned Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.merge = True
        args.convert = True

    if not (args.merge or args.convert):
        parser.print_help()
        return

    config = load_yaml(args.config)
    run_export(config, merge=args.merge, convert=args.convert)


if __name__ == "__main__":
    main()
