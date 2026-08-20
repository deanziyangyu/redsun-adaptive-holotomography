# redsun-adaptive-holotomography

Reusable acquisition and processing components for the Adaptive Multimodal
Holotomographic Microscope.

> [!IMPORTANT]
> This repository is in its Phase 1 scaffold. It currently provides public
> contracts, configuration profiles, versioned service envelopes, and a
> deterministic journal-backed simulated run lifecycle. Camera, stage,
> illumination, reconstruction, and GUI integrations are planned and are not
> operational yet.

The distribution imports as `redsun_aht` and installs the `aht` command.

## Development

The project targets Python 3.12+ and pins the first compatibility baseline to
RedSun 0.11.0.

```console
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy
```

Run a hardware-free lifecycle smoke test with:

```console
uv run aht simulate --journal run-events.jsonl
```

The simulated command creates an append-only event journal, completes one run,
and replays the journal to verify the terminal state. It does not construct or
contact hardware.

## Current package boundaries

- `domain`: immutable IDs, frames, poses, calibration records, and run events.
- `protocols.py`: public detector, stage, motion-validation, processing, and
  remote-reconstruction extension seams.
- `transport`: schema-versioned MsgPack service envelopes.
- `storage`: immutable run manifests and append-only event journals.
- `configurations`: inspectable hardware-free build/run factories.
- `providers.py` and `streams.py`: typed container keys and canonical event
  stream names.

See `docs/component-disposition.md` for the initial depend/upstream/port/rewrite
ledger.
