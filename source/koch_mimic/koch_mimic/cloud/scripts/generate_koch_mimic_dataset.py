"""Generate additional Koch demonstrations with Isaac Lab Mimic."""

from __future__ import annotations

import sys

from ._isaaclab_mimic_runner import run_isaaclab_mimic_script


def main(argv: list[str] | None = None) -> None:
    run_isaaclab_mimic_script(
        "generate_dataset.py",
        sys.argv[1:] if argv is None else argv,
        patch_generation_loop=True,
    )


if __name__ == "__main__":
    main()
