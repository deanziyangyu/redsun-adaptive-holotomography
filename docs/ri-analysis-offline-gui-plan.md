# RI analysis in the offline reconstruction GUI

Status: planned after headless RI analysis and standalone plotting are characterized.

## Goal

Expose a completed or newly run RI analysis in the existing hardware-free
Napari processing application without adding acquisition devices, processing
flyers, or real-time sample-search behavior.

## Composition

Add a dedicated `RiAnalysisPresenter` and `RiAnalysisWidget` beside the
existing `OfflineProcessingPresenter` and `ProcessingWidget`.  The
`launch_processing_gui` composition will add it as a second right-hand dock
named **AHT RI Analysis** in the same Napari window.  Keeping a separate dock
prevents the raw-run reconstruction form from acquiring TIFF-specific fields
or lifecycle state.

The presenter will accept a reconstruction TIFF, a fresh analysis directory,
and a typed `RiAnalysisConfig`.  It will preflight the TIFF with `load_ri_tiff`,
run `analyze_ri_tiff` in a `QRunnable`, verify the manifest, and return paths
plus display-ready arrays.  It will never construct hardware devices or submit
a `ProcessingJob`.

## UI behavior

- Source and output selectors show TIFF shape, decoded RI range, and physical
  Z/Y/X spacing after validation.
- Controls expose the candidate strategy, Sobel percentile, RI threshold mode,
  morphology size, inclusive label voxel-count bounds, physical grid size, and
  shared histogram limits.  Advanced oversized-component refinement remains
  collapsed by default.
- Run/preview/copy-command actions mirror the existing offline widget.  Run is
  disabled while a worker is active and cancellation remains a future addition.
- After a verified run or bundle load, Napari receives RI and Sobel images with
  physical scale plus candidate ROI, selected ROI, unfiltered labels, and
  retained labels as appropriately typed layers.
- Label, grid, and label-grid tables use pageable `QTableView` models backed by
  columnar Zarr arrays. The canonical label-grid table records each retained
  label's voxel count, physical volume, RI moments (including mean, median and
  standard deviation), percentiles, and shared-bin histogram in every
  intersecting cubic grid cell. Selecting a label highlights its layer ID;
  selecting a grid row draws its physical bounding box. The UI does not load
  all per-group voxel values solely to render a table.
- Explicit **Create violin plot** actions call the standalone
  `plot_grid_sampled_label_violins` helper and open the saved PNG in Napari or
  the platform viewer.  It groups each sample at one x-axis tick, renders one
  violin per retained label, and plots one point for every intersecting physical
  grid cell. Plotting stays optional and outside analysis execution.

## Validation and promotion

- Add presenter tests for TIFF preflight, command rendering, manifest loading,
  layer scale, and a rejected malformed/occupied output.
- Add Qt tests for control-to-config mapping, disabled-running state, table
  selection/highlighting, and plot-action error reporting when the `analysis`
  extra is absent.
- Replay the supplied 60×600×600 donor TIFF, compare GUI and headless bundle
  checksums/configuration, and confirm no hardware-profile imports occur.
- Do not promote this dock into sample-search or live acquisition profiles
  until separate latency, cancellation, and decision-policy requirements are
  defined.
