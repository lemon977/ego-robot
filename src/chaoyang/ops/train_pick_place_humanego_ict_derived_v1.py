"""Separate derived-supervision training candidate, not real robot controls."""
from chaoyang.humanego_ict.derived_training_cli import main as _main


def main():
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
