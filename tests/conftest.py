"""Shared test configuration.

The suite is headless: no test may construct a Qt ``QApplication`` or a napari
viewer. GUI behaviour is qualified on the host, not here.

Temporary directories
---------------------
pytest 9 creates its ``--basetemp`` and each numbered run directory with
``mode=0o700``. On Windows that becomes a restrictive DACL, so a later process
can neither traverse nor delete the tree and cleanup fails with
``PermissionError: [WinError 5]``. The repository runner also exports
``scripts/dsh/sitecustomize.py`` for the same reason, but a direct
``python -m pytest`` invocation does not go through the runner, so the patch is
repeated here before pytest creates any directory. It only relaxes the mode
argument; it grants no privilege and changes no repository content.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")

if sys.platform == "win32":
    _mkdir = os.mkdir

    def _permissive_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        """Create a directory with a permissive mode, ignoring ``mode``."""
        _mkdir(path, 0o777, dir_fd=dir_fd)

    os.mkdir = _permissive_mkdir
