"""Optional Qt/Napari hardware-free processing view."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from qtpy import QtCore, QtWidgets
from redsun.view import ViewPosition
from redsun.view.qt import QtView

from redsun_aht.domain import MultiSliceAcquisitionMode
from redsun_aht.presenter import (
    OFFLINE_PROCESSING_PRESENTER,
    OfflineProcessingPresenter,
    ProcessingViewResult,
)
from redsun_aht.processing import (
    MultiLayerReconstructionConfig,
    MultiLayerSolveConfig,
    OfflineMultiLayerJobRequest,
    OfflineTiledJobRequest,
    QuantitativeTileConfig,
    TilingConfig,
)

if TYPE_CHECKING:
    from redsun.virtual import VirtualContainer

    from redsun_aht.processing import OfflineProcessingPlan


class _RunnerSignals(QtCore.QObject):
    succeeded = QtCore.Signal(object)  # type: ignore[attr-defined]
    failed = QtCore.Signal(str)  # type: ignore[attr-defined]


class _ProcessingRunner(QtCore.QRunnable):
    def __init__(
        self, presenter: OfflineProcessingPresenter, plan: OfflineProcessingPlan
    ) -> None:
        super().__init__()
        self.presenter = presenter
        self.plan = plan
        self.signals = _RunnerSignals()

    def run(self) -> None:
        """Execute on the Qt thread pool and report a bounded result/error."""
        try:
            result = self.presenter.execute(self.plan)
        except Exception as error:
            self.signals.failed.emit(f"{type(error).__name__}: {error}")
        else:
            self.signals.succeeded.emit(result)


class ProcessingWidget(QtView):
    """Small processing-only form with no acquisition device dependencies."""

    def __init__(
        self,
        name: str | OfflineProcessingPresenter = "processing",
        /,
        *,
        presenter: OfflineProcessingPresenter | None = None,
        viewer: Any | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if isinstance(name, OfflineProcessingPresenter):
            presenter = name
            name = "processing"
        super().__init__(name)
        if parent is not None:
            self.setParent(parent)
        self._presenter = presenter
        self.viewer = viewer
        thread_pool = QtCore.QThreadPool.globalInstance()
        if thread_pool is None:  # pragma: no cover - Qt runtime invariant
            raise RuntimeError("Qt global thread pool is unavailable")
        self._thread_pool: QtCore.QThreadPool = thread_pool
        self._runner: _ProcessingRunner | None = None
        self._build_ui()

    @property
    def view_position(self) -> ViewPosition:
        """Place the AHT processing controls beside the image workspace."""
        return ViewPosition.RIGHT

    @property
    def presenter(self) -> OfflineProcessingPresenter:
        """Return the injected RedSun presenter."""
        if self._presenter is None:
            raise RuntimeError(
                "processing presenter is available after container build"
            )
        return self._presenter

    def inject_dependencies(self, container: VirtualContainer) -> None:
        """Resolve the processing presenter from the RedSun container."""
        if self._presenter is None:
            self._presenter = container.require(OFFLINE_PROCESSING_PRESENTER)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        notice = QtWidgets.QLabel(
            "Offline processing only — this profile constructs no acquisition hardware."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)

        form = QtWidgets.QFormLayout()
        self.input_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit()
        form.addRow("Verified run bundle", self._path_row(self.input_edit, True))
        form.addRow("Product directory", self._path_row(self.output_edit, False))
        self.gpu_check = QtWidgets.QCheckBox("Use CuPy GPU projection")
        form.addRow("Backend", self.gpu_check)
        self.gpu_id = QtWidgets.QSpinBox()
        self.gpu_id.setRange(0, 255)
        form.addRow("GPU ID", self.gpu_id)
        self.reservation_mib = QtWidgets.QSpinBox()
        self.reservation_mib.setRange(1, 1024 * 1024)
        self.reservation_mib.setValue(256)
        form.addRow("VRAM reservation (MiB)", self.reservation_mib)
        layout.addLayout(form)
        layout.addWidget(self._tiled_group())
        layout.addWidget(self._multilayer_group())

        actions = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Validate / Preview")
        self.copy_button = QtWidgets.QPushButton("Copy command")
        self.run_button = QtWidgets.QPushButton("Run")
        actions.addWidget(self.preview_button)
        actions.addWidget(self.copy_button)
        actions.addWidget(self.run_button)
        layout.addLayout(actions)

        self.command_edit = QtWidgets.QPlainTextEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setMaximumBlockCount(20)
        layout.addWidget(self.command_edit)
        self.status_label = QtWidgets.QLabel("Idle")
        layout.addWidget(self.status_label)

        self.gpu_check.toggled.connect(self._update_gpu_controls)
        self.preview_button.clicked.connect(self.preview_command)
        self.copy_button.clicked.connect(self.copy_command)
        self.run_button.clicked.connect(self.run_processing)
        self._update_gpu_controls(False)

    def _tiled_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Optional tiled quantitative job")
        group.setCheckable(True)
        group.setChecked(False)
        self.tiled_group = group
        form = QtWidgets.QFormLayout(group)
        self.tiled_model = QtWidgets.QComboBox()
        self.tiled_model.addItems(("dpct", "qobt"))
        self.tiled_gpu_check = QtWidgets.QCheckBox("Use CuPy quantitative worker")
        self.tiled_gpu_id = QtWidgets.QSpinBox()
        self.tiled_gpu_id.setRange(0, 255)
        self.tiled_gpu_reservation = QtWidgets.QSpinBox()
        self.tiled_gpu_reservation.setRange(1, 1024 * 1024)
        self.tiled_gpu_reservation.setValue(256)
        self.tiled_detector = QtWidgets.QLineEdit("dhm")
        self.background_detector = QtWidgets.QLineEdit()
        self.background_z = QtWidgets.QLineEdit()
        self.fixed_median = QtWidgets.QCheckBox(
            "Fixed median over this run's four-shot Z stack"
        )
        form.addRow("Model", self.tiled_model)
        form.addRow("Backend", self.tiled_gpu_check)
        form.addRow("Tiled GPU ID", self.tiled_gpu_id)
        form.addRow("Tiled VRAM reservation (MiB)", self.tiled_gpu_reservation)
        form.addRow("Sample detector", self.tiled_detector)
        form.addRow("Background detector (optional)", self.background_detector)
        form.addRow("Background Z index (optional)", self.background_z)
        form.addRow("Background source", self.fixed_median)

        self.wavelength = self._positive_float(0.625, 6)
        self.numerical_aperture = self._positive_float(0.8, 6)
        self.pixel_size = self._positive_float(0.108333, 9)
        self.pixel_size_z = self._positive_float(1.0, 9)
        self.medium_index = self._positive_float(1.0, 6)
        self.sample_index = self._positive_float(1.4, 6)
        self.source_azimuth = QtWidgets.QLineEdit("0, 90, 180, 270")
        self.source_elevation = self._float(45.0, -360.0, 360.0, 6)
        self.source_sigma = self._positive_float(0.5, 6)
        form.addRow("Wavelength (um)", self.wavelength)
        form.addRow("Numerical aperture", self.numerical_aperture)
        form.addRow("Pixel size XY (um)", self.pixel_size)
        form.addRow("Pixel size Z (um)", self.pixel_size_z)
        form.addRow("Medium refractive index", self.medium_index)
        form.addRow("Sample refractive index", self.sample_index)
        form.addRow("Four source azimuths (deg)", self.source_azimuth)
        form.addRow("Source elevation (deg)", self.source_elevation)
        form.addRow("Source sigma", self.source_sigma)

        self.background_division = QtWidgets.QComboBox()
        self.background_division.addItems(
            ("background_over_sample", "sample_over_background")
        )
        self.regularization_real = self._positive_float(5e-5, 9)
        self.regularization_imaginary = self._positive_float(5e-5, 9)
        self.regularization_scalar = self._positive_float(1e-2, 9)
        self.axial_window = QtWidgets.QComboBox()
        self.axial_window.addItems(("hamming", "rectangular", "none"))
        form.addRow("Background division", self.background_division)
        form.addRow("Real regularization", self.regularization_real)
        form.addRow("Imaginary regularization", self.regularization_imaginary)
        form.addRow("Scalar regularization", self.regularization_scalar)
        form.addRow("Axial window", self.axial_window)

        self.tile_mode = QtWidgets.QComboBox()
        self.tile_mode.addItems(("auto", "off"))
        self.tile_voxel_budget = QtWidgets.QSpinBox()
        self.tile_voxel_budget.setRange(1, 2_147_483_647)
        self.tile_voxel_budget.setValue(512 * 512 * 100)
        self.tile_max_shape = QtWidgets.QLineEdit("768, 768")
        self.tile_min_shape = QtWidgets.QLineEdit("256, 256")
        self.tile_alignment = QtWidgets.QSpinBox()
        self.tile_alignment.setRange(1, 1_000_000)
        self.tile_alignment.setValue(32)
        self.tile_overlap_fraction = self._float(0.2, 0.0, 0.499999, 6)
        self.tile_pad_mode = QtWidgets.QComboBox()
        self.tile_pad_mode.addItems(("reflect", "edge"))
        self.tile_shape = QtWidgets.QLineEdit()
        self.tile_overlap = QtWidgets.QLineEdit()
        form.addRow("Tile mode", self.tile_mode)
        form.addRow("Voxel budget", self.tile_voxel_budget)
        form.addRow("Maximum tile Y, X", self.tile_max_shape)
        form.addRow("Minimum tile Y, X", self.tile_min_shape)
        form.addRow("Alignment (px)", self.tile_alignment)
        form.addRow("Overlap fraction", self.tile_overlap_fraction)
        form.addRow("Padding", self.tile_pad_mode)
        form.addRow("Explicit tile Y, X (optional)", self.tile_shape)
        form.addRow("Explicit overlap Y, X (optional)", self.tile_overlap)
        self.tiled_gpu_check.toggled.connect(self._update_tiled_gpu_controls)
        self.fixed_median.toggled.connect(self._update_fixed_median_controls)
        self._update_tiled_gpu_controls(False)
        return group

    def _update_tiled_gpu_controls(self, enabled: bool) -> None:
        self.tiled_gpu_id.setEnabled(enabled)
        self.tiled_gpu_reservation.setEnabled(enabled)

    def _update_fixed_median_controls(self, enabled: bool) -> None:
        """Keep mutually exclusive external-background controls unambiguous."""
        self.background_detector.setEnabled(not enabled)
        self.background_z.setEnabled(not enabled)
        if enabled:
            self.background_detector.clear()
            self.background_z.clear()

    def _multilayer_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Optional nonlinear multi-layer job")
        group.setCheckable(True)
        group.setChecked(False)
        self.multilayer_group = group
        form = QtWidgets.QFormLayout(group)
        self.multilayer_model = QtWidgets.QComboBox()
        self.multilayer_model.addItems(("multi_born", "multislice"))
        self.multilayer_gpu_check = QtWidgets.QCheckBox("Use CuPy nonlinear worker")
        self.multilayer_gpu_id = QtWidgets.QSpinBox()
        self.multilayer_gpu_id.setRange(0, 255)
        self.multilayer_gpu_reservation = QtWidgets.QSpinBox()
        self.multilayer_gpu_reservation.setRange(1, 1024 * 1024)
        self.multilayer_gpu_reservation.setValue(512)
        self.multilayer_detector = QtWidgets.QLineEdit("dhm")
        self.multilayer_background_detector = QtWidgets.QLineEdit()
        self.multilayer_background_z = QtWidgets.QLineEdit()
        form.addRow("Model", self.multilayer_model)
        form.addRow("Backend", self.multilayer_gpu_check)
        form.addRow("Multi-layer GPU ID", self.multilayer_gpu_id)
        form.addRow(
            "Multi-layer VRAM reservation (MiB)",
            self.multilayer_gpu_reservation,
        )
        form.addRow("Sample detector", self.multilayer_detector)
        form.addRow(
            "Background detector (optional)", self.multilayer_background_detector
        )
        form.addRow("Background Z index (optional)", self.multilayer_background_z)

        self.multilayer_depth = QtWidgets.QSpinBox()
        self.multilayer_depth.setRange(1, 1_000_000)
        self.multilayer_depth.setValue(32)
        self.multilayer_voxel_size = QtWidgets.QLineEdit("0.25, 0.108333, 0.108333")
        self.multilayer_wavelength = self._positive_float(0.625, 6)
        self.multilayer_na = self._positive_float(0.8, 6)
        self.multilayer_medium_index = self._positive_float(1.33, 6)
        self.multilayer_illumination_na = self._positive_float(0.45, 6)
        self.multilayer_acquisition_mode = QtWidgets.QComboBox()
        self.multilayer_acquisition_mode.addItems(
            [mode.value for mode in MultiSliceAcquisitionMode]
        )
        self.multilayer_source_z = QtWidgets.QSpinBox()
        self.multilayer_source_z.setRange(0, 1_000_000)
        self.multilayer_padding = QtWidgets.QLineEdit("0, 0")
        self.multilayer_defocus = self._float(0.0, -1_000_000.0, 1_000_000.0, 6)
        self.multilayer_focus_offsets = QtWidgets.QLineEdit("0")
        self.multilayer_skip_shots = QtWidgets.QLineEdit()
        self.multilayer_normalization = QtWidgets.QComboBox()
        self.multilayer_normalization.addItems(("shot_mean", "background", "none"))
        self.multilayer_memory_budget = QtWidgets.QSpinBox()
        self.multilayer_memory_budget.setRange(1, 1024 * 1024)
        self.multilayer_memory_budget.setValue(512)
        form.addRow("Depth layers", self.multilayer_depth)
        form.addRow("Voxel size Z, Y, X (um)", self.multilayer_voxel_size)
        form.addRow("Wavelength (um)", self.multilayer_wavelength)
        form.addRow("Detection NA", self.multilayer_na)
        form.addRow("Medium refractive index", self.multilayer_medium_index)
        form.addRow("Illumination NA", self.multilayer_illumination_na)
        form.addRow("Acquisition mode", self.multilayer_acquisition_mode)
        form.addRow("Source acquisition Z index", self.multilayer_source_z)
        form.addRow("Padding Y, X", self.multilayer_padding)
        form.addRow("Defocus (um)", self.multilayer_defocus)
        form.addRow("MSBP focus offsets (slices)", self.multilayer_focus_offsets)
        form.addRow("MSBP skipped shots (zero-based)", self.multilayer_skip_shots)
        form.addRow("Normalization", self.multilayer_normalization)
        form.addRow("CPU memory budget (MiB)", self.multilayer_memory_budget)

        self.multilayer_iterations = QtWidgets.QSpinBox()
        self.multilayer_iterations.setRange(1, 1_000_000)
        self.multilayer_iterations.setValue(5)
        self.multilayer_step = self._float(0.0, 0.0, 1_000_000.0, 9)
        self.multilayer_optimizer = QtWidgets.QComboBox()
        self.multilayer_optimizer.addItems(("fista", "gradient_descent"))
        self.multilayer_measurement_domain = QtWidgets.QComboBox()
        self.multilayer_measurement_domain.addItems(("intensity", "amplitude", "field"))
        self.multilayer_l2 = self._float(0.0, 0.0, 1_000_000.0, 9)
        self.multilayer_tv = self._float(0.0, 0.0, 1_000_000.0, 9)
        self.multilayer_tv_iterations = QtWidgets.QSpinBox()
        self.multilayer_tv_iterations.setRange(1, 1_000_000)
        self.multilayer_tv_iterations.setValue(15)
        self.multilayer_early_stopping = self._float(0.0, 0.0, 1_000_000.0, 9)
        self.multilayer_subtract_first_slice_mean = QtWidgets.QCheckBox(
            "Subtract first-slice mean"
        )
        self.multilayer_recover_pupil = QtWidgets.QCheckBox("Recover pupil")
        self.multilayer_pupil_step = self._float(0.0, 0.0, 1_000_000.0, 9)
        self.multilayer_pupil_method = QtWidgets.QComboBox()
        self.multilayer_pupil_method.addItems(("gradient", "gauss_newton"))
        form.addRow("Iterations", self.multilayer_iterations)
        form.addRow("Step size (0 = model default)", self.multilayer_step)
        form.addRow("Optimizer", self.multilayer_optimizer)
        form.addRow("Measurement domain", self.multilayer_measurement_domain)
        form.addRow("L2 weight", self.multilayer_l2)
        form.addRow("TV weight", self.multilayer_tv)
        form.addRow("TV iterations", self.multilayer_tv_iterations)
        form.addRow(
            "Early stop relative change (0 = off)",
            self.multilayer_early_stopping,
        )
        form.addRow(
            "MSBP baseline correction", self.multilayer_subtract_first_slice_mean
        )
        form.addRow("Pupil", self.multilayer_recover_pupil)
        form.addRow("Pupil step size", self.multilayer_pupil_step)
        form.addRow("Pupil update", self.multilayer_pupil_method)
        self.multilayer_gpu_check.toggled.connect(self._update_multilayer_gpu_controls)
        self._update_multilayer_gpu_controls(False)
        return group

    def _update_multilayer_gpu_controls(self, enabled: bool) -> None:
        self.multilayer_gpu_id.setEnabled(enabled)
        self.multilayer_gpu_reservation.setEnabled(enabled)

    @staticmethod
    def _float(
        value: float, minimum: float, maximum: float, decimals: int
    ) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setValue(value)
        return box

    @classmethod
    def _positive_float(cls, value: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
        return cls._float(value, 10 ** (-decimals), 1_000_000.0, decimals)

    @staticmethod
    def _numbers(text: str, count: int, label: str) -> tuple[float, ...]:
        try:
            values = tuple(float(part.strip()) for part in text.split(","))
        except ValueError as error:
            raise ValueError(
                f"{label} must contain numeric comma-separated values"
            ) from error
        if len(values) != count:
            raise ValueError(f"{label} must contain exactly {count} values")
        return values

    @classmethod
    def _integer_pair(
        cls, text: str, label: str, *, optional: bool = False
    ) -> tuple[int, int] | None:
        if optional and not text.strip():
            return None
        values = cls._numbers(text, 2, label)
        if any(not value.is_integer() for value in values):
            raise ValueError(f"{label} must contain integers")
        return int(values[0]), int(values[1])

    def _tiled_request(self) -> OfflineTiledJobRequest | None:
        if not self.tiled_group.isChecked():
            return None
        background = self.background_detector.text().strip() or None
        background_z_text = self.background_z.text().strip()
        try:
            background_z = None if not background_z_text else int(background_z_text)
        except ValueError as error:
            raise ValueError("background Z index must be an integer") from error
        optical = QuantitativeTileConfig(
            model=cast("Literal['dpct', 'qobt']", self.tiled_model.currentText()),
            wavelength_um=self.wavelength.value(),
            numerical_aperture=self.numerical_aperture.value(),
            pixel_size_um=self.pixel_size.value(),
            pixel_size_z_um=self.pixel_size_z.value(),
            refractive_index_medium=self.medium_index.value(),
            refractive_index_sample=self.sample_index.value(),
            source_azimuth_deg=self._numbers(
                self.source_azimuth.text(), 4, "source azimuths"
            ),
            source_elevation_deg=self.source_elevation.value(),
            source_sigma=self.source_sigma.value(),
            background_division=cast(
                "Literal['background_over_sample', 'sample_over_background']",
                self.background_division.currentText(),
            ),
            regularization_real=self.regularization_real.value(),
            regularization_imaginary=self.regularization_imaginary.value(),
            regularization_scalar=self.regularization_scalar.value(),
            axial_window=cast(
                "Literal['hamming', 'rectangular', 'none']",
                self.axial_window.currentText(),
            ),
        )
        max_shape = self._integer_pair(self.tile_max_shape.text(), "maximum tile shape")
        min_shape = self._integer_pair(self.tile_min_shape.text(), "minimum tile shape")
        assert max_shape is not None and min_shape is not None
        tiling = TilingConfig(
            mode=cast("Literal['auto', 'off']", self.tile_mode.currentText()),
            voxel_budget=self.tile_voxel_budget.value(),
            max_shape_yx=max_shape,
            min_shape_yx=min_shape,
            alignment_px=self.tile_alignment.value(),
            overlap_fraction=self.tile_overlap_fraction.value(),
            pad_mode=cast(
                "Literal['reflect', 'edge']", self.tile_pad_mode.currentText()
            ),
            explicit_shape_yx=self._integer_pair(
                self.tile_shape.text(), "explicit tile shape", optional=True
            ),
            explicit_overlap_yx=self._integer_pair(
                self.tile_overlap.text(), "explicit tile overlap", optional=True
            ),
        )
        return OfflineTiledJobRequest(
            self.tiled_detector.text().strip(),
            optical,
            tiling,
            background_detector_id=background,
            background_z_index=background_z,
            fixed_median_from_sample=self.fixed_median.isChecked(),
            backend="cupy" if self.tiled_gpu_check.isChecked() else "numpy",
            gpu_id=(
                self.tiled_gpu_id.value() if self.tiled_gpu_check.isChecked() else None
            ),
            gpu_reservation_bytes=self.tiled_gpu_reservation.value() * 1024 * 1024,
        )

    def _multilayer_request(self) -> OfflineMultiLayerJobRequest | None:
        if not self.multilayer_group.isChecked():
            return None
        voxel_size = self._numbers(
            self.multilayer_voxel_size.text(), 3, "multi-layer voxel size"
        )
        padding = self._integer_pair(
            self.multilayer_padding.text(), "multi-layer padding"
        )
        assert padding is not None
        background = self.multilayer_background_detector.text().strip() or None
        background_z_text = self.multilayer_background_z.text().strip()
        try:
            background_z = None if not background_z_text else int(background_z_text)
        except ValueError as error:
            raise ValueError(
                "multi-layer background Z index must be an integer"
            ) from error
        step_value = self.multilayer_step.value()
        focus_offsets = tuple(
            float(part.strip())
            for part in self.multilayer_focus_offsets.text().split(",")
            if part.strip()
        )
        try:
            skip_shots = tuple(
                int(part.strip())
                for part in self.multilayer_skip_shots.text().split(",")
                if part.strip()
            )
        except ValueError as error:
            raise ValueError(
                "MSBP skipped shots must be comma-separated integers"
            ) from error
        early_stopping = self.multilayer_early_stopping.value()
        solve = MultiLayerSolveConfig(
            max_iterations=self.multilayer_iterations.value(),
            step_size=None if step_value == 0 else step_value,
            optimizer=cast(
                "Literal['fista', 'gradient_descent']",
                self.multilayer_optimizer.currentText(),
            ),
            measurement_domain=cast(
                "Literal['intensity', 'amplitude', 'field']",
                self.multilayer_measurement_domain.currentText(),
            ),
            l2_weight=self.multilayer_l2.value(),
            tv_weight=self.multilayer_tv.value(),
            tv_iterations=self.multilayer_tv_iterations.value(),
            recover_pupil=self.multilayer_recover_pupil.isChecked(),
            pupil_step_size=self.multilayer_pupil_step.value(),
            pupil_update_method=cast(
                "Literal['gradient', 'gauss_newton']",
                self.multilayer_pupil_method.currentText(),
            ),
            early_stopping_relative=(None if early_stopping == 0 else early_stopping),
            subtract_first_slice_mean=(
                self.multilayer_subtract_first_slice_mean.isChecked()
            ),
        )
        reconstruction = MultiLayerReconstructionConfig(
            model=cast(
                "Literal['multi_born', 'multislice']",
                self.multilayer_model.currentText(),
            ),
            depth_layers=self.multilayer_depth.value(),
            voxel_size_zyx_um=cast("tuple[float, float, float]", voxel_size),
            wavelength_um=self.multilayer_wavelength.value(),
            numerical_aperture=self.multilayer_na.value(),
            refractive_index_medium=self.multilayer_medium_index.value(),
            illumination_na=self.multilayer_illumination_na.value(),
            acquisition_mode=MultiSliceAcquisitionMode(
                self.multilayer_acquisition_mode.currentText()
            ),
            source_z_index=self.multilayer_source_z.value(),
            padding_yx=padding,
            defocus_um=self.multilayer_defocus.value(),
            focus_offsets_slices=focus_offsets,
            skip_shots=skip_shots,
            normalization=cast(
                "Literal['none', 'shot_mean', 'background']",
                self.multilayer_normalization.currentText(),
            ),
            solve=solve,
            memory_budget_bytes=self.multilayer_memory_budget.value() * 1024 * 1024,
        )
        use_gpu = self.multilayer_gpu_check.isChecked()
        return OfflineMultiLayerJobRequest(
            self.multilayer_detector.text().strip(),
            reconstruction,
            background_detector_id=background,
            background_z_index=background_z,
            backend="cupy" if use_gpu else "numpy",
            gpu_id=self.multilayer_gpu_id.value() if use_gpu else None,
            gpu_reservation_bytes=(
                self.multilayer_gpu_reservation.value() * 1024 * 1024
            ),
        )

    def _path_row(self, edit: QtWidgets.QLineEdit, existing: bool) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit)
        button = QtWidgets.QPushButton("Browse")
        button.clicked.connect(lambda: self._browse(edit, existing))
        layout.addWidget(button)
        return container

    def _browse(self, edit: QtWidgets.QLineEdit, existing: bool) -> None:
        if existing:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select verified DPCT run bundle", edit.text()
            )
        else:
            selected = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select product parent directory", edit.text()
            )
        if selected:
            edit.setText(selected)

    def _update_gpu_controls(self, enabled: bool) -> None:
        self.gpu_id.setEnabled(enabled)
        self.reservation_mib.setEnabled(enabled)

    def _resolve(self) -> OfflineProcessingPlan:
        return self.presenter.resolve(
            self.input_edit.text(),
            self.output_edit.text(),
            gpu_id=self.gpu_id.value() if self.gpu_check.isChecked() else None,
            gpu_reservation_mib=self.reservation_mib.value(),
            tiled=self._tiled_request(),
            multilayer=self._multilayer_request(),
        )

    def preview_command(self) -> None:
        """Validate inputs, create typed jobs, and display an equivalent CLI."""
        try:
            plan = self._resolve()
            command = self.presenter.preview_command(plan)
        except Exception as error:
            self._show_error(error)
            return
        self.command_edit.setPlainText(command)
        self.status_label.setText(
            f"Validated {len(plan.jobs)} jobs for {plan.source_run_id}"
        )

    def copy_command(self) -> None:
        """Validate and copy the platform-quoted command to the clipboard."""
        self.preview_command()
        command = self.command_edit.toPlainText()
        if command:
            clipboard = QtWidgets.QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(command)

    def run_processing(self) -> None:
        """Start detached solvers on the Qt worker pool."""
        if self._runner is not None:
            return
        try:
            plan = self._resolve()
        except Exception as error:
            self._show_error(error)
            return
        runner = _ProcessingRunner(self.presenter, plan)
        runner.signals.succeeded.connect(self._on_succeeded)
        runner.signals.failed.connect(self._on_failed)
        self._runner = runner
        self.run_button.setEnabled(False)
        self.status_label.setText("Processing…")
        self._thread_pool.start(runner)

    def _on_succeeded(self, value: object) -> None:
        result = cast("ProcessingViewResult", value)
        for solver_id, array in result.products.items():
            if self.viewer is not None:
                self.viewer.add_image(array, name=f"{result.source_run_id}:{solver_id}")
        lines = [f"{solver_id}: {uri}" for solver_id, uri in result.output_uris.items()]
        lines.extend(
            f"{solver_id}: failed {message}"
            for solver_id, message in result.failures.items()
        )
        self.command_edit.setPlainText("\n".join(lines))
        self.status_label.setText(
            f"Completed {len(result.products)} products; "
            f"{len(result.failures)} failures"
        )
        self._finish_runner()

    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"Failed: {message}")
        self._finish_runner()

    def _show_error(self, error: Exception) -> None:
        self.status_label.setText(f"Invalid: {error}")
        self.command_edit.clear()

    def _finish_runner(self) -> None:
        self._runner = None
        self.run_button.setEnabled(True)


def launch_processing_gui(
    presenter: OfflineProcessingPresenter | None = None,
    *,
    input_root: str | None = None,
    output_root: str | None = None,
    run_event_loop: bool = True,
) -> tuple[Any, ProcessingWidget]:
    """Create the RedSun-composed Napari application and optionally run it."""
    import napari
    from redsun.containers import declare_presenter, declare_view
    from redsun.qt import QtAppContainer

    from redsun_aht.configurations.profiles import profile_path

    active_presenter = presenter or OfflineProcessingPresenter()
    viewer = napari.Viewer(title="RedSun AHT Offline Processing")

    class AHTProcessingContainer(QtAppContainer, config=profile_path("process-gui")):
        processing = declare_presenter(
            OfflineProcessingPresenter, executable=active_presenter.executable
        )
        processing_view = declare_view(ProcessingWidget, viewer=viewer)

    container = AHTProcessingContainer(
        session="AHT offline processing", frontend="pyqt"
    ).build()
    widget = cast("ProcessingWidget", container.views["processing_view"])
    widget._redsun_container = container  # type: ignore[attr-defined]
    if input_root is not None:
        widget.input_edit.setText(input_root)
    if output_root is not None:
        widget.output_edit.setText(output_root)
    viewer.window.add_dock_widget(widget, name="AHT Processing", area="right")
    if run_event_loop:
        try:
            napari.run()
        finally:
            container.shutdown()
    return viewer, widget


__all__ = ["ProcessingWidget", "launch_processing_gui"]
