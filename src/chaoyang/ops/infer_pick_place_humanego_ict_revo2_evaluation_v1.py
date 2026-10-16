"""Finite offline evaluation; no robot or optimizer calls."""


def main(argv=None):
    from chaoyang.humanego_ict.evaluation_runtime_v1 import main as run
    return run(argv)
