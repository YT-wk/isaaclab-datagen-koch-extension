"""Shared layered YAML configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from .constants import CLOUD_PROFILE, CONFIG_FILENAMES, LOCAL_PROFILE


class RuntimeConfigError(RuntimeError):
    """Raised when a runtime configuration is missing or invalid."""


RuntimeProfile = str


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for one profile."""

    profile: RuntimeProfile
    data: dict[str, Any]
    repo_root: Path
    defaults_path: Path
    user_local_path: Path
    user_example_path: Path
    overlay_path: Path | None = None


_ACTIVE_CONFIGS: dict[RuntimeProfile, RuntimeConfig] = {}


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up until a `configs` directory exists."""
    search_start = (start or Path(__file__)).resolve()
    candidates = [search_start, *search_start.parents]
    for candidate in candidates:
        if candidate.is_dir():
            root = candidate
        else:
            root = candidate.parent
        if (root / "configs").is_dir() and (root / "source" / "koch_mimic").is_dir():
            return root
    raise RuntimeConfigError("Failed to locate the repository root that contains `configs/` and `source/koch_mimic/`.")


def _profile_file_paths(profile: RuntimeProfile, repo_root: Path) -> tuple[Path, Path, Path]:
    if profile not in CONFIG_FILENAMES:
        raise RuntimeConfigError(f"Unsupported runtime profile: {profile!r}")
    filenames = CONFIG_FILENAMES[profile]
    config_dir = repo_root / "configs"
    return (
        config_dir / filenames["defaults"],
        config_dir / filenames["example"],
        config_dir / filenames["user_local"],
    )


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RuntimeConfigError(f"Expected a YAML mapping in {path}, got {type(data).__name__}.")
    return data


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    normalized = {}
    for key, value in overrides.items():
        if value is not None:
            normalized[key] = value
    return normalized


def load_runtime_config(
    profile: RuntimeProfile,
    *,
    overlay_path: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    require_user_local: bool = True,
    repo_root: Path | None = None,
) -> RuntimeConfig:
    """Load `defaults -> user.local -> optional overlay -> CLI overrides`."""
    root = find_repo_root(repo_root or Path(__file__))
    defaults_path, example_path, user_local_path = _profile_file_paths(profile, root)
    if not defaults_path.is_file():
        raise RuntimeConfigError(f"Missing defaults configuration: {defaults_path}")

    merged = _read_yaml_mapping(defaults_path)
    if require_user_local and not user_local_path.is_file():
        raise RuntimeConfigError(
            f"Missing required user configuration: {user_local_path}\n"
            f"Create it by copying: {example_path}"
        )
    if user_local_path.is_file():
        merged = _deep_merge(merged, _read_yaml_mapping(user_local_path))

    resolved_overlay_path: Path | None = None
    if overlay_path:
        resolved_overlay_path = Path(overlay_path).expanduser().resolve()
        if not resolved_overlay_path.is_file():
            raise RuntimeConfigError(f"Overlay config file does not exist: {resolved_overlay_path}")
        merged = _deep_merge(merged, _read_yaml_mapping(resolved_overlay_path))

    merged = _deep_merge(merged, _normalize_overrides(cli_overrides))
    return RuntimeConfig(
        profile=profile,
        data=merged,
        repo_root=root,
        defaults_path=defaults_path,
        user_local_path=user_local_path,
        user_example_path=example_path,
        overlay_path=resolved_overlay_path,
    )


def activate_runtime_config(
    profile: RuntimeProfile,
    *,
    overlay_path: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    require_user_local: bool = True,
) -> RuntimeConfig:
    """Load and cache the runtime configuration for later package access."""
    config = load_runtime_config(
        profile,
        overlay_path=overlay_path,
        cli_overrides=cli_overrides,
        require_user_local=require_user_local,
    )
    _ACTIVE_CONFIGS[profile] = config
    return config


def get_active_runtime_config(profile: RuntimeProfile, *, require_user_local: bool = False) -> RuntimeConfig:
    """Return the active configuration, or lazily load defaults when needed."""
    config = _ACTIVE_CONFIGS.get(profile)
    if config is not None:
        return config
    return load_runtime_config(profile, require_user_local=require_user_local)


def resolve_config_path(path_value: str | None, config: RuntimeConfig | None = None) -> str | None:
    """Resolve a config path relative to the repository root."""
    if path_value is None or path_value == "":
        return path_value
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    runtime_config = config or get_active_runtime_config(CLOUD_PROFILE, require_user_local=False)
    return str((runtime_config.repo_root / candidate).resolve())


def get_config_section(config: RuntimeConfig, *keys: str, default: Any = None) -> Any:
    """Read a nested key path from a runtime config."""
    value: Any = config.data
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        if key not in value:
            return default
        value = value[key]
    return value


def as_tuple(value: Any) -> tuple[Any, ...] | None:
    """Convert lists to tuples for configclass fields."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise RuntimeConfigError(f"Expected a list or tuple, got {type(value).__name__}.")


def option_was_provided(argv: list[str], option: str) -> bool:
    """Best-effort check whether a CLI option was supplied."""
    return option in argv or any(token.startswith(f"{option}=") for token in argv)


def profile_name_is_valid(profile: RuntimeProfile) -> bool:
    return profile in (CLOUD_PROFILE, LOCAL_PROFILE)

