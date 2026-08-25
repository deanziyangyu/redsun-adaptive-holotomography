"""Tiled catalog registration and verified replay without base-install coupling."""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

from redsun_aht.acquisition import DocumentRecord
from redsun_aht.domain.models import thaw_json
from redsun_aht.processing import observations_from_dpct_bundle
from redsun_aht.storage import DocumentJournal, DpctZarrReplay

if TYPE_CHECKING:
    from redsun_aht.domain import Observation, ProcessingResult
    from redsun_aht.domain.models import JsonValue

_ASSET_ROOT_KEY = "aht-assets"


class TiledUnavailableError(RuntimeError):
    """Raised when the optional catalog runtime is not installed."""


class CatalogConflictError(RuntimeError):
    """Raised when a registration would replace an existing catalog node."""


class CatalogSelectionError(RuntimeError):
    """Raised when a catalog query has no unique, valid run selection."""


@dataclass(frozen=True, slots=True)
class CatalogRegistration:
    """One completed Bluesky-run and Zarr-asset catalog registration."""

    catalog_run_uid: str
    source_run_id: str
    document_count: int
    asset_paths: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze catalog path identities and validate completion."""
        if (
            not self.catalog_run_uid
            or not self.source_run_id
            or self.document_count <= 0
            or "raw" not in self.asset_paths
        ):
            raise ValueError("catalog registration fields are incomplete")
        object.__setattr__(
            self, "asset_paths", MappingProxyType(dict(self.asset_paths))
        )


@dataclass(frozen=True, slots=True)
class CatalogRunSelection:
    """Minimal metadata required to replay one Tiled-selected processing run."""

    catalog_run_uid: str
    source_run_id: str
    source_bundle_uri: str
    processing_mode: str
    started_at: float

    def __post_init__(self) -> None:
        """Validate stable run and source identities."""
        if (
            not self.catalog_run_uid
            or not self.source_run_id
            or not self.source_bundle_uri
            or not self.processing_mode
            or self.started_at < 0
        ):
            raise ValueError("catalog run selection fields are incomplete")


async def register_processing_run(
    client: Any,
    journal_path: Path,
    results: Mapping[str, ProcessingResult],
) -> CatalogRegistration:
    """Register one complete journal plus verified raw and derived Zarr assets."""
    TiledWriter, _, register, _ = _tiled_api()
    records = DocumentJournal(journal_path).read()
    start = _single_document(records, "start")
    _single_document(records, "stop")
    catalog_run_uid = _required_string(start.document, "uid")
    source_run_id = _required_string(start.document, "aht_source_run_id")
    source_bundle_uri = _required_string(start.document, "aht_source_bundle_uri")
    if client.get(catalog_run_uid) is not None:
        raise CatalogConflictError(
            f"Tiled processing run already exists: {catalog_run_uid}"
        )
    if not results or any(
        result.run_id != source_run_id for result in results.values()
    ):
        raise ValueError("catalog results do not match the source run")

    source_bundle_path = _file_path(source_bundle_uri)
    DpctZarrReplay(source_bundle_path).verify()
    writer = TiledWriter(client, batch_size=1, validate=False)
    for record in records:
        payload = thaw_json(record.document)
        if not isinstance(payload, dict):  # pragma: no cover - record invariant
            raise TypeError("Bluesky document payload must be a dictionary")
        writer(record.name, payload)
    if client.get(catalog_run_uid) is None:
        raise RuntimeError("Tiled did not publish the completed processing run")

    assets_root = _get_or_create_container(
        client,
        _ASSET_ROOT_KEY,
        metadata={"aht_catalog_schema": 1, "purpose": "raw-and-derived-assets"},
    )
    source_key = _source_key(source_run_id)
    source_assets = _get_or_create_container(
        assets_root,
        source_key,
        metadata={
            "aht_source_run_id": source_run_id,
            "aht_source_bundle_uri": source_bundle_uri,
        },
    )
    asset_paths: dict[str, str] = {}
    raw_path = source_bundle_path / "data.ome.zarr"
    raw_asset = source_assets.get("raw")
    if raw_asset is None:
        await register(
            source_assets,
            raw_path,
            walkers=[_stop_after_store_registration],
            key_from_filename=lambda _: "raw",
            overwrite=False,
        )
        raw_asset = source_assets.get("raw")
    if raw_asset is None:
        raise RuntimeError("Tiled did not register the raw OME-Zarr asset")
    raw_asset.update_metadata(
        {
            "aht_catalog": {
                "role": "raw",
                "source_run_id": source_run_id,
                "source_bundle_uri": source_bundle_uri,
            }
        }
    )
    asset_paths["raw"] = f"{_ASSET_ROOT_KEY}/{source_key}/raw"

    processing_key = f"processing-{catalog_run_uid}"
    if source_assets.get(processing_key) is not None:
        raise CatalogConflictError(
            f"Tiled processing assets already exist: {processing_key}"
        )
    processing_assets = source_assets.create_container(
        key=processing_key,
        metadata={
            "aht_catalog_run_uid": catalog_run_uid,
            "aht_source_run_id": source_run_id,
        },
    )
    for solver_id, result in sorted(results.items()):
        product_path = _file_path(result.output.uri)
        await register(
            processing_assets,
            product_path,
            walkers=[_stop_after_store_registration],
            key_from_filename=lambda _, key=solver_id: key,
            overwrite=False,
        )
        asset = processing_assets.get(solver_id)
        if asset is None:
            raise RuntimeError(f"Tiled did not register product for {solver_id}")
        asset.update_metadata(
            {
                "aht_catalog": {
                    "role": "derived",
                    "catalog_run_uid": catalog_run_uid,
                    "source_run_id": source_run_id,
                    "solver_id": solver_id,
                    "result_id": result.result_id,
                    "output_uri": result.output.uri,
                    "output_checksum": result.output.checksum,
                }
            }
        )
        asset_paths[solver_id] = (
            f"{_ASSET_ROOT_KEY}/{source_key}/{processing_key}/{solver_id}"
        )
    return CatalogRegistration(
        catalog_run_uid,
        source_run_id,
        len(records),
        asset_paths,
    )


def search_processing_runs(
    client: Any, source_run_id: str
) -> tuple[CatalogRunSelection, ...]:
    """Find processing runs by source-run metadata, oldest first."""
    if not source_run_id:
        raise ValueError("catalog source run identity cannot be empty")
    _, Key, _, _ = _tiled_api()
    matches = client.search(Key("start.aht_source_run_id") == source_run_id)
    selections = tuple(
        _selection_from_run(str(uid), matches[uid]) for uid in tuple(matches)
    )
    return tuple(
        sorted(selections, key=lambda item: (item.started_at, item.catalog_run_uid))
    )


def select_processing_run(
    client: Any,
    source_run_id: str,
    *,
    catalog_run_uid: str | None = None,
) -> CatalogRunSelection:
    """Select one exact catalog run, rejecting missing or ambiguous searches."""
    if catalog_run_uid is not None:
        run = client.get(catalog_run_uid)
        if run is None:
            raise CatalogSelectionError(f"Tiled run does not exist: {catalog_run_uid}")
        selection = _selection_from_run(catalog_run_uid, run)
        if selection.source_run_id != source_run_id:
            raise CatalogSelectionError("Tiled run belongs to another source run")
        return selection
    selections = search_processing_runs(client, source_run_id)
    if len(selections) != 1:
        raise CatalogSelectionError(
            f"Tiled source run selection is not unique: {len(selections)} matches"
        )
    return selections[0]


def observations_from_catalog(
    client: Any,
    source_run_id: str,
    *,
    catalog_run_uid: str | None = None,
) -> tuple[Observation, ...]:
    """Resolve a Tiled-selected source bundle through the verified replay reader."""
    selection = select_processing_run(
        client,
        source_run_id,
        catalog_run_uid=catalog_run_uid,
    )
    return observations_from_dpct_bundle(_file_path(selection.source_bundle_uri))


def catalog_documents(client: Any, catalog_run_uid: str) -> tuple[DocumentRecord, ...]:
    """Export a Tiled run as semantically reconstructed Bluesky documents."""
    run = client.get(catalog_run_uid)
    if run is None or not hasattr(run, "documents"):
        raise CatalogSelectionError(
            f"Tiled run cannot export documents: {catalog_run_uid}"
        )
    records: list[DocumentRecord] = []
    for name, document in run.documents():
        records.append(DocumentRecord(name, cast("Mapping[str, JsonValue]", document)))
    if not records or records[0].name != "start" or records[-1].name != "stop":
        raise CatalogSelectionError("Tiled document export is incomplete")
    return tuple(records)


def connect_tiled(uri: str) -> Any:
    """Connect to a configured Tiled server without persisting credentials."""
    if not uri:
        raise ValueError("Tiled server URI cannot be empty")
    _, _, _, from_uri = _tiled_api()
    return from_uri(uri, remember_me=False)


def _single_document(records: tuple[DocumentRecord, ...], name: str) -> DocumentRecord:
    matches = tuple(record for record in records if record.name == name)
    if len(matches) != 1:
        raise ValueError(f"processing journal requires one {name} document")
    return matches[0]


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"processing document field is missing: {key}")
    return value


def _selection_from_run(catalog_run_uid: str, run: Any) -> CatalogRunSelection:
    metadata = run.metadata
    start = metadata.get("start") if isinstance(metadata, Mapping) else None
    if not isinstance(start, Mapping):
        raise CatalogSelectionError("Tiled run metadata lacks a start document")
    uid = _required_string(start, "uid")
    if uid != catalog_run_uid:
        raise CatalogSelectionError("Tiled run key and start UID differ")
    source_run_id = _required_string(start, "aht_source_run_id")
    source_bundle_uri = _required_string(start, "aht_source_bundle_uri")
    processing_mode = _required_string(start, "aht_processing_mode")
    started_at = start.get("time")
    if not isinstance(started_at, int | float) or isinstance(started_at, bool):
        raise CatalogSelectionError("Tiled run start time is invalid")
    _file_path(source_bundle_uri)
    return CatalogRunSelection(
        catalog_run_uid,
        source_run_id,
        source_bundle_uri,
        processing_mode,
        float(started_at),
    )


def _get_or_create_container(
    parent: Any, key: str, *, metadata: Mapping[str, JsonValue]
) -> Any:
    container = parent.get(key)
    if container is None:
        container = parent.create_container(key=key, metadata=dict(metadata))
    return container


def _source_key(source_run_id: str) -> str:
    digest = hashlib.sha256(source_run_id.encode()).hexdigest()[:20]
    return f"source-{digest}"


def _file_path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file":
        raise ValueError("catalog assets must use file URIs")
    path_text = unquote(parsed.path)
    if os.name == "nt" and path_text.startswith("/"):
        path_text = path_text[1:]
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError("catalog file URI did not resolve to an absolute path")
    return path


async def _stop_after_store_registration(
    _node: Any,
    _path: Path,
    _files: list[Path],
    _directories: list[Path],
    _settings: Any,
) -> tuple[list[Path], list[Path]]:
    """Prevent Tiled 0.2 from descending after registering a Zarr store."""
    return [], []


def _tiled_api() -> tuple[Any, Any, Any, Any]:
    try:
        writer_module = importlib.import_module(
            "bluesky_tiled_plugins.writing.tiled_writer"
        )
        query_module = importlib.import_module("tiled.queries")
        register_module = importlib.import_module("tiled.client.register")
        client_module = importlib.import_module("tiled.client")
    except ImportError as error:
        raise TiledUnavailableError(
            "Tiled catalog support is unavailable; install the catalog-tiled extra"
        ) from error
    return (
        writer_module.TiledWriter,
        query_module.Key,
        register_module.register,
        client_module.from_uri,
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
