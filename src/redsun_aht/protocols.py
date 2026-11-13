"""Structural device contracts for the live-stream plan helpers.

Adapted from ``redsun-planner`` (package ``sample_search``) at revision
``c0eff86``, file ``src/planner/protocols.py``, Apache-2.0 licensed; copyright
the ``redsun-planner`` contributors. The member docstrings keep the donor's
meaning.

Changed for :mod:`redsun_aht`:

- :class:`LiveFlyer` also declares ``stage``/``unstage``, because the live
  helpers unstage the detectors they were given, and ``stop``, because the
  RunEngine stops the devices of an interrupted run. It is also readable,
  because a stream declared with ``collect=False`` is described with
  ``describe`` rather than ``describe_collect``.
- ``XYMotor`` is not carried over: the donor's live-stream helper never needs
  it. A later scan plan may add its own axis contract instead of importing this
  module.

Notes
-----
:class:`LiveFlyer` is the contract the live plans *accept*. An ophyd-async
``StandardDetector`` has ``stage``/``unstage`` and ``read``/``describe`` but no
``stop``; a device selector that must accept one unchanged should test a
narrower protocol, while a camera that owns a ``stop`` can be a whole
:class:`LiveFlyer`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from bluesky.protocols import (
    Collectable,
    Flyable,
    Preparable,
    Readable,
    Stageable,
    Stoppable,
    WritesStreamAssets,
)


@runtime_checkable
class LiveFlyer(
    Readable[Any],
    Stageable,
    Preparable,
    Flyable,
    Collectable,
    WritesStreamAssets,
    Stoppable,
    Protocol,
):
    """A Bluesky flyer that also exposes its live image buffer to a viewer."""

    @property
    def name(self) -> str:
        """Return the flyer's name."""
        ...

    @property
    def buffer(self) -> Any:
        """Return the live image buffer a viewer reads."""
        ...

    def get_index(self) -> Any:
        """Return the index of the newest frame in the buffer."""
        ...


__all__ = ["LiveFlyer"]
