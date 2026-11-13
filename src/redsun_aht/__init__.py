"""Adaptive holotomography acquisition application.

The package is being rebuilt on the IX83 imaging bundle's basis: PVA-only
device services, an editable camera and stage, a live stream routed through
Bluesky Events, and a four-pattern DPCT plan driven by the MM-compatible
Arduino illumination service.

Nothing in this module imports the retired Channel Access application. The
archived implementation lives in ``.archive/aht-20261103-0919ee5``.
"""

from ._version import __version__

__all__ = ["__version__"]
