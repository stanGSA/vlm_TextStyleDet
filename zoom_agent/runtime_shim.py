"""Runtime compatibility bridge to read-only E:\\textstyleDet\\textstyleDet_rn.

1. Enforces ``sys.dont_write_bytecode = True`` before any upstream import so zero
   ``.pyc`` files are written into ``textstyleDet_rn``.
2. Provides an import-only Windows ``fcntl`` stub whose ``flock``/``fcntl``/``ioctl``
   explicitly raise ``NotImplementedError`` if ever called (preventing silent fake
   locks if someone accidentally invokes training lock functions in the same process).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True

DEFAULT_RN_ROOT = Path(r"E:\textstyleDet\textstyleDet_rn")


def _forbidden_unix_lock(*_args, **_kwargs):
    raise NotImplementedError(
        "Windows fcntl shim in zoom_agent.runtime_shim is strictly import-only for "
        "frozen evaluation. Process/file locking via fcntl is forbidden on Windows."
    )


def ensure_textstyle_rn_importable(rn_root: Path | str = DEFAULT_RN_ROOT) -> Path:
    """Register import-only Windows fcntl guard and add ``textstyleDet_rn`` to ``sys.path``."""
    sys.dont_write_bytecode = True

    if sys.platform == "win32" and "fcntl" not in sys.modules:
        fcntl_stub = types.ModuleType("fcntl")
        fcntl_stub.LOCK_EX = 2
        fcntl_stub.LOCK_SH = 1
        fcntl_stub.LOCK_NB = 4
        fcntl_stub.LOCK_UN = 8
        fcntl_stub.flock = _forbidden_unix_lock
        fcntl_stub.fcntl = _forbidden_unix_lock
        fcntl_stub.ioctl = _forbidden_unix_lock
        sys.modules["fcntl"] = fcntl_stub

    resolved = Path(rn_root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"textstyleDet_rn root not found: {resolved}")

    resolved_str = str(resolved)
    if resolved_str not in sys.path:
        sys.path.insert(0, resolved_str)
    return resolved
