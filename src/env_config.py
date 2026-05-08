from __future__ import annotations

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_NAME = ".env"


def load_env(env_path: Path | None = None) -> Path | None:
    path = env_path or PROJECT_ROOT / ENV_FILE_NAME
    if not path.exists():
        return None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)

    return path


def get_env_value(name: str, default: str) -> str:
    load_env()
    return os.environ.get(name, default)


def get_path(name: str, default_relative: str | Path) -> Path:
    raw_value = get_env_value(name, str(default_relative))
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def resolve_input_file(
    user_input: str | None,
    env_var_name: str,
    default_relative: str | Path,
) -> Path:
    if user_input:
        path = Path(str(user_input).replace("\\", "/")).expanduser()
        if path.exists() and path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"Khong tim thay file theo input: {path}")

    path = get_path(env_var_name, default_relative)
    if path.exists() and path.is_file():
        return path.resolve()

    raise FileNotFoundError(
        f"Khong tim thay file cho bien {env_var_name}: {path}\n"
        f"Hay cap nhat .env hoac truyen duong dan qua CLI."
    )


def resolve_directory(
    user_input: str | None,
    env_var_name: str,
    default_relative: str | Path,
) -> Path:
    if user_input:
        return Path(str(user_input).replace("\\", "/")).expanduser().resolve(strict=False)
    return get_path(env_var_name, default_relative)
