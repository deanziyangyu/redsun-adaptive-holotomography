# RedSun multidimensional storage PR request note

> **Superseded:** Maintainer feedback established that live-acquisition storage
> should be owned by services/devices rather than a replacement RedSun storage
> shim. Use
> [`redsun-multidimensional-storage-issue-note.md`](redsun-multidimensional-storage-issue-note.md)
> as the current issue draft. This document remains as design history for the
> experimental `03009a5` contract.

This is a draft request for review, not an opened issue or pull request. It
describes the next potential RedSun upstream change prototyped on
`deanziyangyu/redsun:feat/pre-upstream` at commit `03009a5`.

## Suggested title

RFC: add transactional multidimensional storage contracts

## Proposed request

### Summary

RedSun's current storage interface is optimized for sequential 2-D frame
streams. Applications that acquire into named multidimensional coordinates
cannot tell a backend where a frame belongs, carry write provenance through
the storage boundary, await a durable-write receipt, or explicitly complete
or abort a dataset.

This proposal adds an independent, runtime-checkable multidimensional storage
protocol and small immutable request/receipt models. It is intended as a
generic RedSun contract, not as an AHT- or Mimir-specific storage schema.

The prototype is deliberately additive: it does not extend or alter the
existing `OpenStore` API, and existing sequential backends and applications
continue unchanged.

### Motivation

The existing `OpenStore.write(data_key, frame)` call does not encode named
coordinates. `FrameSink.put()` confirms queue admission rather than backend
durability. Those semantics are insufficient for acquisitions that require:

- deterministic placement across detector, pattern, scan, channel, or other
  application-defined axes;
- provenance supplied with each write;
- an acknowledgement issued only after the backend's durability guarantee;
- a persisted checksum that can be compared with the source frame; and
- explicit successful completion or failure/abort publication.

AHT's DPCT Zarr path is one concrete reproducer, but its axis vocabulary and
bundle format should remain outside RedSun. Mimir feature parity is not a
goal. The goal is to make RedSun capable of owning the underlying generic
storage boundary.

### Proposed public contract

The prototype exports the following from `redsun.storage`:

- `FramePlacement(axes, index)` validates a non-empty, unique set of named
  axes and a same-length tuple of non-negative integer indices.
- `IndexedWrite(data_key, frame, placement, source_checksum, context)` carries
  a NumPy frame, deterministic placement, source checksum, and opaque backend
  context.
- `PersistedWrite(data_key, placement, uri, checksum)` identifies the durable
  result and its backend checksum.
- `MultidimensionalOpenStore` defines asynchronous `write_indexed()`,
  `complete()`, and `abort()` operations.

Illustrative use:

```python
placement = FramePlacement(
    axes=("pattern", "scan"),
    index=(pattern_ordinal, scan_ordinal),
)
request = IndexedWrite(
    data_key="camera",
    frame=frame,
    placement=placement,
    source_checksum=source_checksum,
    context={"exposure_id": exposure_id},
)
receipt = await store.write_indexed(request)
if receipt.checksum != request.source_checksum:
    raise RuntimeError("persisted frame checksum mismatch")
await store.complete()
```

The intended semantic requirement is that `write_indexed()` must not return a
receipt before the backend has met its documented durability guarantee. Queue
admission alone is not a persisted-write receipt.

### Reconnaissance

At the time of this prototype:

- canonical RedSun `main` on the v0.12.2 line retained the sequential storage
  interface;
- RedSun `feat/experimental-di` changed composition and lifecycle facilities,
  but did not introduce a multidimensional storage contract; and
- canonical redsun-mimir `main` and `feat/mmcore-upstream-devices` continued to
  use the sequential `BaseStorage`/`FrameSink` path.

There is therefore no emerging multidimensional RedSun interface to reuse.
These branches should be inspected again immediately before preparing a PR.

### Prototype validation

For commit `03009a5`:

- 8 focused multidimensional-storage tests passed;
- the full RedSun suite passed: 570 tests;
- Ruff, Mypy across 113 source files, and `git diff --check` passed; and
- AHT's 26 existing DPCT Zarr tests passed while importing the public contract
  from the editable RedSun checkout.

The AHT result is a compatibility/regression check only. AHT is not yet routed
through this protocol, so it is not evidence of end-to-end contract
sufficiency.

### Open design questions

The prototype should be reviewed as an RFC until these points are resolved:

1. Should multidimensional storage remain a separate protocol, or become a
   compatible evolution of `OpenStore`?
2. Should axes, extents, dtype, shape, and chunk layout be declared once in a
   stream schema rather than repeating axis names on every write?
3. What precise durability level must a `PersistedWrite` receipt guarantee,
   and how should a backend test or document it?
4. Should checksum identity include an algorithm field instead of requiring
   lowercase SHA-256?
5. Should abort receive a Python exception, serializable failure metadata, or
   both?
6. What `StorageIO` factory/open contract should construct a multidimensional
   store?
7. How should `BaseStorage` and `FrameSink` expose persisted receipts without
   confusing them with producer-queue acknowledgement?
8. How should resource/datum documents represent array path, placement, shape,
   dtype, chunks, and per-write context?
9. Must the first merged PR include a reference backend and enforced
   complete-versus-abort state machine?

The current `context` mapping is only shallowly immutable, and the protocol
documents rather than enforces lifecycle state. These details should not be
presented as finalized API decisions.

### Recommended delivery

Prefer either combining the contract with a small in-memory/reference backend,
or keeping the first PR explicitly marked as a draft RFC. A protocol-only
merge risks establishing semantics before they can be exercised.

A staged implementation could be:

1. Finalize request, placement, receipt, lifecycle, and stream-schema models;
   add a reference transactional backend and state-transition tests.
2. Add factory integration plus receipt-aware `BaseStorage`/`FrameSink`
   plumbing while preserving the sequential API.
3. Add a production Zarr implementation and event-model resource/datum
   representation, including durability, replay, collision, and abort tests.
4. Route AHT's DPCT store through the accepted interface only after its current
   checksum, journal, manifest, and incomplete-bundle tests pass unchanged in
   meaning.

### Compatibility and non-goals

- Existing `OpenStore`, `BaseStorage`, `FrameSink`, and acquire-zarr behavior
  must remain source-compatible unless a separately reviewed migration is
  approved.
- This proposal does not make RedSun match Mimir feature-for-feature.
- It does not place AHT-specific DPCT axes or manifests in RedSun.
- It does not weaken AHT's existing acknowledgement or recovery guarantees.
- It does not include the EPICS service-device changes.

### Branch preparation

Do not open `feat/pre-upstream` directly as one PR. The storage commit is
stacked on the EPICS candidate (`1a1879f`), while the two changes require
independent review.

For a standalone storage PR, create a clean branch from the then-current
canonical `redsun-acquisition/redsun` main branch and cherry-pick or rework the
storage commit `03009a5`. Alternatively, stack it only after the EPICS PR has
merged. Re-run branch reconnaissance, the full test/type/lint suite, and
coverage before publishing the review-ready branch.

### Acceptance checklist for an implementation PR

- Named multidimensional placement is validated deterministically.
- A write receipt has an explicit, testable durability meaning.
- Complete and abort are mutually exclusive and idempotence is defined.
- Failure cannot publish a successful completion marker.
- Backend creation is available through a public RedSun factory boundary.
- Resource/datum metadata describes physical array placement and layout.
- Existing sequential users remain compatible.
- A reference or Zarr backend covers success, collision, corruption, replay,
  incomplete acquisition, completion, and abort behavior.
- AHT can adopt the interface without side-channel placement or provenance and
  without reducing its present durability guarantees.
