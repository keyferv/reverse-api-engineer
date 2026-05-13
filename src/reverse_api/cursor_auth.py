"""Resolve Cursor API keys for the Node bridge and Python checks."""

from __future__ import annotations

import os
from pathlib import Path


def get_cursor_api_key() -> str:
    """Return the Cursor API key from the process environment or a key file.

    Only **exported** shell variables are visible here. If ``echo $CURSOR_API_KEY``
    works but this returns empty, use ``export CURSOR_API_KEY=...`` or set
    ``CURSOR_API_KEY_FILE`` to a file whose first non-comment line is the key.
    """
    v = (os.environ.get("CURSOR_API_KEY") or "").strip()
    if v:
        return v
    path = (os.environ.get("CURSOR_API_KEY_FILE") or "").strip()
    if not path:
        return ""
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return ""
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                return t
    except OSError:
        return ""
    return ""
