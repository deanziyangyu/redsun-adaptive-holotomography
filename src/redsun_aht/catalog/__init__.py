"""Optional Tiled registration, query, and replay adapters."""

from .tiled import (
    CatalogConflictError,
    CatalogRegistration,
    CatalogRunSelection,
    CatalogSelectionError,
    TiledUnavailableError,
    catalog_documents,
    connect_tiled,
    observations_from_catalog,
    register_processing_run,
    search_processing_runs,
    select_processing_run,
)

__all__ = [
    "CatalogConflictError",
    "CatalogRegistration",
    "CatalogRunSelection",
    "CatalogSelectionError",
    "TiledUnavailableError",
    "catalog_documents",
    "connect_tiled",
    "observations_from_catalog",
    "register_processing_run",
    "search_processing_runs",
    "select_processing_run",
]
