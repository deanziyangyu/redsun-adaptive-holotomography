"""Force permissive directory modes for tools started by the DSH test runner.

pytest 9 creates its ``--basetemp`` and each numbered run directory with
``mode=0o700`` (``_pytest/tmpdir.py``). On Windows that becomes a restrictive
DACL: a later process under a different or sandboxed token can neither traverse
nor delete the tree, and cleanup fails with ``PermissionError: [WinError 5]``
after every test has passed.

CPython's ``site`` machinery imports this module when its directory is first on
``PYTHONPATH``. It replaces ``os.mkdir`` with a version that always requests
``0o777``: a no-op on POSIX, and on Windows it avoids the DACL case.
``pathlib.Path.mkdir`` calls ``os.mkdir``, so both entry points are covered.

It changes no repository content and grants no privilege; drop the directory
from ``PYTHONPATH`` to disable it.
"""

import os
import sys

if sys.platform == "win32":
    _mkdir = os.mkdir

    def mkdir(path, mode=0o777, *, dir_fd=None):
        """Create a directory with a permissive mode, ignoring ``mode``."""
        return _mkdir(path, 0o777, dir_fd=dir_fd)

    os.mkdir = mkdir
