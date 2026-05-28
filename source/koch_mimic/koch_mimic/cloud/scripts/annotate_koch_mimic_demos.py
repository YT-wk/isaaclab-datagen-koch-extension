"""Annotate Koch demonstration datasets for Isaac Lab Mimic generation."""

from __future__ import annotations

import sys

from ._isaaclab_mimic_runner import run_isaaclab_mimic_script


def main(argv: list[str] | None = None) -> None:
    run_isaaclab_mimic_script(
        "annotate_demos.py",
        sys.argv[1:] if argv is None else argv,
        configure_annotation_actions=True,
    )


if __name__ == "__main__":
    main()
