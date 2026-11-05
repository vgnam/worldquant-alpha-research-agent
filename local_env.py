from __future__ import annotations

import os
from pathlib import Path


def load_local_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    # ``utf-8-sig``, not ``utf-8``: an editor (and PowerShell's ``Set-Content
    # -Encoding utf8``) can add a BOM, and with a plain utf-8 read the first
    # key silently becomes "\ufeffWQB_EMAIL" — so every later lookup of
    # ``WQB_EMAIL`` misses and the run fails with "missing credentials" while
    # the value sits right there in the file.
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
