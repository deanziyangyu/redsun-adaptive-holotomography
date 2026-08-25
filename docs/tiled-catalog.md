# Tiled catalog and replay

The optional catalog layer registers one completed processing-flyer journal as
a Bluesky run and exposes its raw and derived Zarr stores through Tiled. Tiled
is an access and discovery layer here. The fsynced document journal, immutable
manifests, and OME-Zarr/Zarr stores remain the durable records.

## Runtime and server contract

Install the pinned client, server, and Bluesky integration packages with:

```console
uv sync --extra catalog-tiled --locked
```

The Tiled server must include every source and product root in its readable
storage allowlist. It must also enable the Bluesky run exporter if clients need
to reconstruct documents with `catalog_documents` or `run.documents()`:

```yaml
media_types:
  BlueskyRun:
    application/json-seq: bluesky_tiled_plugins.exporters:json_seq_exporter
```

Authentication remains a Tiled deployment concern. `connect_tiled` delegates
to Tiled's client and does not persist credentials locally.

## Registration

Run the hardware-free flyer replay and register it in one command:

```console
uv run aht process-flyer-replay \
  --input C:/data/run-001 \
  --output C:/data/products/run-001 \
  --tiled-uri http://localhost:8000
```

Registration is accepted only when the journal contains exactly one start and
one stop document, its source metadata is complete, every processing result
belongs to the same source run, and the DPCT bundle passes manifest, journal,
coverage, and frame-checksum verification. A catalog run UID or processing
asset path is never overwritten. Repeating the same registration raises
`CatalogConflictError`.

The resulting catalog layout is:

```text
/<processing-run-uid>/
  aht_processing_progress/
  aht_processing_result/
/aht-assets/
  source-<sha256-prefix>/
    raw/                              # external data.ome.zarr
    processing-<processing-run-uid>/
      mean-projection/                # external product.ome.zarr
      quality-metrics/                # external product.zarr
```

Asset metadata records role, source run, processing run, solver, result,
output URI, and output checksum. The source container key hashes the full
source-run ID so arbitrary run IDs do not become catalog paths.

## Query and replay

`search_processing_runs(client, source_run_id)` searches the nested
`start.aht_source_run_id` metadata and returns selections in start-time order.
`select_processing_run` requires one unique match unless an exact processing
run UID is supplied; missing, mismatched, and ambiguous selections are errors.

`observations_from_catalog` resolves the selected `file:` source-bundle URI and
then calls the same verified DPCT replay reader used by headless processing. It
does not trust catalog metadata as proof of data integrity. Raw and product
arrays remain directly sliceable through their registered Tiled nodes.

The integration test starts a real local Tiled HTTP server, registers two
independent processing runs, queries by source identity, exports Bluesky
documents, reads a raw detector array through Tiled, and compares it with the
checksum-verified local replay.
