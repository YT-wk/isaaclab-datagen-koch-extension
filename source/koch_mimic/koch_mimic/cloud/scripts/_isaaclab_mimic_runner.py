"""Helpers for running Isaac Lab Mimic scripts with this external task registered."""

from __future__ import annotations

from pathlib import Path
import sys

from koch_mimic.shared.configuration import find_repo_root


def _extract_config_arg(argv: list[str]) -> tuple[str | None, list[str]]:
    config_path: str | None = None
    filtered: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--config":
            if index + 1 >= len(argv):
                raise ValueError("--config requires a YAML path")
            config_path = argv[index + 1]
            index += 2
            continue
        if token.startswith("--config="):
            config_path = token.split("=", 1)[1]
            index += 1
            continue
        filtered.append(token)
        index += 1
    return config_path, filtered


def _find_isaaclab_mimic_script(script_name: str) -> Path:
    relative = Path("scripts") / "imitation_learning" / "isaaclab_mimic" / script_name
    repo_root = find_repo_root()
    candidates = [
        repo_root / "external" / "IsaacLab" / relative,
        Path.home() / "isaaclab" / relative,
        Path("/workspace/isaaclab") / relative,
        Path("/root/isaaclab") / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find Isaac Lab Mimic script {script_name!r}. Searched:\n{searched}")


def run_isaaclab_mimic_script(
    script_name: str,
    argv: list[str] | None = None,
    *,
    configure_annotation_actions: bool = False,
    patch_generation_loop: bool = False,
) -> None:
    """Execute an Isaac Lab Mimic CLI script after injecting Koch task registration."""
    original_argv = list(sys.argv[1:] if argv is None else argv)
    config_path, forwarded_argv = _extract_config_arg(original_argv)
    script_path = _find_isaaclab_mimic_script(script_name)

    source = script_path.read_text(encoding="utf-8")
    marker = "import isaaclab_tasks  # noqa: F401"
    injection = (
        f"{marker}\n"
        "from koch_mimic.shared.configuration import activate_runtime_config as _koch_activate_runtime_config\n"
        "from koch_mimic.shared.constants import CLOUD_PROFILE as _KOCH_CLOUD_PROFILE\n"
        "_koch_activate_runtime_config(\n"
        "    _KOCH_CLOUD_PROFILE,\n"
        "    overlay_path=globals().get('_KOCH_MIMIC_CONFIG_PATH'),\n"
        "    require_user_local=True,\n"
        ")\n"
        "import koch_mimic.cloud.tasks.koch_pick_place  # noqa: F401\n"
    )
    if marker not in source:
        raise RuntimeError(f"Could not find injection point in {script_path}")
    source = source.replace(marker, injection, 1)
    if configure_annotation_actions:
        parse_marker = "env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)"
        parse_replacement = (
            f"{parse_marker}\n"
            "    from koch_mimic.cloud.scripts.mimic_action_compat import (\n"
            "        configure_annotation_action_space_for_dataset as _koch_configure_annotation_action_space,\n"
            "    )\n"
            "    _koch_configure_annotation_action_space(env_cfg, args_cli.input_file)\n"
        )
        if parse_marker not in source:
            raise RuntimeError(f"Could not find annotation action-space injection point in {script_path}")
        source = source.replace(parse_marker, parse_replacement, 1)

    if patch_generation_loop:
        import_marker = "from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config"
        import_replacement = (
            f"{import_marker}\n"
            "from koch_mimic.cloud.scripts.mimic_generation_compat import (\n"
            "    wrap_generation_env_loop as _koch_wrap_generation_env_loop,\n"
            ")\n"
            "env_loop = _koch_wrap_generation_env_loop(env_loop)\n"
        )
        if import_marker not in source:
            raise RuntimeError(f"Could not find generation-loop injection point in {script_path}")
        source = source.replace(import_marker, import_replacement, 1)

    previous_argv = sys.argv
    sys.argv = [str(script_path), *forwarded_argv]
    sys.path.insert(0, str(script_path.parent))
    try:
        exec_globals = {
            "__name__": "__main__",
            "__file__": str(script_path),
            "_KOCH_MIMIC_CONFIG_PATH": config_path,
        }
        exec(compile(source, str(script_path), "exec"), exec_globals)
    finally:
        sys.argv = previous_argv
