from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from redsun_aht.catalog import (
    CatalogConflictError,
    CatalogRegistration,
    CatalogRunSelection,
    CatalogSelectionError,
    catalog_documents,
    connect_tiled,
    observations_from_catalog,
    register_processing_run,
    search_processing_runs,
    select_processing_run,
)
from redsun_aht.configurations import (
    build_dpct_simulation,
    run_processing_flyer_replay,
)
from redsun_aht.processing import observations_from_dpct_bundle
from redsun_aht.storage import DpctZarrReplay

tiled_client = pytest.importorskip("tiled.client")
tiled_server = pytest.importorskip("tiled.server")
tiled_media_types = pytest.importorskip("tiled.media_type_registration")
bluesky_exporters = pytest.importorskip("bluesky_tiled_plugins.exporters")

pytestmark = pytest.mark.catalog


@pytest.fixture
def local_catalog(tmp_path: Path) -> Generator[Any, None, None]:
    tiled_media_types.default_serialization_registry.register(
        "BlueskyRun",
        "application/json-seq",
        bluesky_exporters.json_seq_exporter,
    )
    readable_storage = tmp_path / "catalog-readable"
    readable_storage.mkdir()
    with tiled_server.SimpleTiledServer(
        directory=tmp_path / "catalog-database",
        readable_storage=readable_storage,
    ) as server:
        yield connect_tiled(server.uri)


def _node_at(client: Any, path: str) -> Any:
    node = client
    for segment in path.split("/"):
        node = node[segment]
    return node


def _create_replay(tmp_path: Path, output_name: str) -> tuple[Path, Any]:
    input_root = tmp_path / "catalog-readable" / "input"
    if not input_root.exists():
        asyncio.run(build_dpct_simulation("catalog-run", output_root=input_root).run())
    replay = run_processing_flyer_replay(
        input_root,
        tmp_path / "catalog-readable" / output_name,
    )
    return input_root, replay


def test_tiled_registers_processing_documents_and_verified_zarr_assets(
    tmp_path: Path,
    local_catalog: Any,
) -> None:
    input_root, replay = _create_replay(tmp_path, "processing-one")

    registration = asyncio.run(
        register_processing_run(local_catalog, replay.journal_path, replay.results)
    )

    assert registration.source_run_id == "catalog-run"
    assert registration.document_count == len(replay.documents)
    assert set(registration.asset_paths) == {
        "raw",
        "mean-projection",
        "quality-metrics",
    }
    selected = select_processing_run(local_catalog, "catalog-run")
    assert selected.catalog_run_uid == registration.catalog_run_uid
    assert selected.source_bundle_uri == input_root.resolve().as_uri()
    assert search_processing_runs(local_catalog, "catalog-run") == (selected,)
    assert (
        select_processing_run(
            local_catalog,
            "catalog-run",
            catalog_run_uid=registration.catalog_run_uid,
        )
        == selected
    )
    assert catalog_documents(local_catalog, registration.catalog_run_uid)[0].name == (
        "start"
    )
    assert catalog_documents(local_catalog, registration.catalog_run_uid)[-1].name == (
        "stop"
    )
    assert observations_from_catalog(local_catalog, "catalog-run") == (
        observations_from_dpct_bundle(input_root)
    )

    manifest = DpctZarrReplay(input_root).verify()
    detector_id, array_path = next(iter(manifest.detector_arrays.items()))
    raw_asset = _node_at(local_catalog, registration.asset_paths["raw"])
    tiled_array = _node_at(raw_asset, array_path)
    np.testing.assert_array_equal(
        tiled_array[:],
        DpctZarrReplay(input_root).read_detector(detector_id),
    )
    assert raw_asset.metadata["aht_catalog"]["role"] == "raw"
    mean_asset = _node_at(local_catalog, registration.asset_paths["mean-projection"])
    assert mean_asset.metadata["aht_catalog"]["output_checksum"] == (
        replay.results["mean-projection"].output.checksum
    )

    with pytest.raises(CatalogConflictError, match="already exists"):
        asyncio.run(
            register_processing_run(
                local_catalog,
                replay.journal_path,
                replay.results,
            )
        )


def test_tiled_source_query_rejects_missing_mismatched_and_ambiguous_runs(
    tmp_path: Path,
    local_catalog: Any,
) -> None:
    _, first = _create_replay(tmp_path, "processing-one")
    first_registration = asyncio.run(
        register_processing_run(local_catalog, first.journal_path, first.results)
    )
    _, second = _create_replay(tmp_path, "processing-two")
    second_registration = asyncio.run(
        register_processing_run(local_catalog, second.journal_path, second.results)
    )

    selections = search_processing_runs(local_catalog, "catalog-run")
    assert {item.catalog_run_uid for item in selections} == {
        first_registration.catalog_run_uid,
        second_registration.catalog_run_uid,
    }
    with pytest.raises(CatalogSelectionError, match="not unique"):
        select_processing_run(local_catalog, "catalog-run")
    with pytest.raises(CatalogSelectionError, match="does not exist"):
        select_processing_run(
            local_catalog,
            "catalog-run",
            catalog_run_uid="missing",
        )
    with pytest.raises(CatalogSelectionError, match="another source"):
        select_processing_run(
            local_catalog,
            "other-run",
            catalog_run_uid=first_registration.catalog_run_uid,
        )


def test_catalog_public_values_and_queries_reject_incomplete_inputs(
    tmp_path: Path,
    local_catalog: Any,
) -> None:
    with pytest.raises(ValueError, match="incomplete"):
        CatalogRegistration("", "source", 1, {"raw": "path"})
    with pytest.raises(ValueError, match="incomplete"):
        CatalogRunSelection("uid", "source", "", "replay", 0)
    with pytest.raises(ValueError, match="cannot be empty"):
        search_processing_runs(local_catalog, "")
    with pytest.raises(ValueError, match="cannot be empty"):
        connect_tiled("")
    with pytest.raises(CatalogSelectionError, match="cannot export"):
        catalog_documents(local_catalog, "missing")

    empty_journal = tmp_path / "empty.jsonl"
    empty_journal.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="one start"):
        asyncio.run(register_processing_run(local_catalog, empty_journal, {}))
