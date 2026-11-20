"""Domain-independent failure contracts shared by scientific processors.

These types describe processing outcomes only. They carry no device, store,
service or framework identity, so an offline caller can catch them without
importing the AHT bundle.
"""

from __future__ import annotations


class ProcessingCancelled(RuntimeError):
    """Raised cooperatively when a processing job is cancelled.

    The type lives in the scientific package so tiled and iterative kernels can
    signal cooperative cancellation without importing the bundle. The historical
    ``redsun_aht.processing.solvers.ProcessingCancelled`` name remains the same
    object through an explicit re-export.
    """


__all__ = ["ProcessingCancelled"]
