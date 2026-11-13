"""Tests for the acquisition page: presenter, view, wiring and notices.

The page is exercised headlessly: the presenter's plan map, its specification
and its annotations are inspected, the session declarations are read without
building the session, and no ``QApplication``, napari viewer or widget is
constructed. The missing-device rule is checked through the pure helpers the
view installs its control from.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Protocol,
    cast,
    get_args,
    get_type_hints,
    runtime_checkable,
)

import pytest
import yaml
from bluesky.utils import MsgGenerator
from redsun import AsPresenter, Layer, Session
from redsun._config import load, validate_session
from redsun.engine.actions import PlanAction
from redsun.path_provider import SessionPathProvider
from redsun.presenter.plan_spec import (
    create_plan_spec,
    resolve_arguments,
)
from redsun.qt import Dock, Qt
from redsun.registry import HasPlans
from redsun.session._declarations import check, read
from redsun.view.qt._device_sequence_edit import DeviceSequenceEdit
from redsun.view.qt._widget_factory import WIDGET_FACTORY_MAP
from redsun.view.qt.builtins import AcquisitionView as BuiltinAcquisitionView

from redsun_aht.configurations import CameraSession, DualCameraSession
from redsun_aht.presenter.acquisition import (
    CAMERA_ROLE_NOTICE,
    AcquisitionPresenter,
    device_protocol,
    missing_device_notice,
    shown_device_parameters,
)
from redsun_aht.presenter.detector import LIVE_STREAM, STREAM, DetectorPresenter
from redsun_aht.protocols import LiveFlyer
from redsun_aht.view.acquisition import (
    AcquisitionView,
    MissingDeviceEdit,
    install_missing_device_control,
    is_unmatched_device,
    make_missing_device_edit,
)


def _acquisition(
    detectors: Mapping[str, Any] | None = None,
) -> tuple[AcquisitionPresenter, DetectorPresenter]:
    """Return a set-up acquisition presenter and the detector presenter it routes to.

    No service, device or engine runs here: the detector presenter holds an
    empty or supplied device mapping and the path provider names no directory.
    """
    detector = DetectorPresenter("det_ctrl", detectors=dict(detectors or {}))
    presenter = AcquisitionPresenter("acq_ctrl", devices={})
    presenter.setup(
        {"det_ctrl": cast("HasPlans", detector)},
        {},
        SessionPathProvider(session="acquisition-page"),
    )
    return presenter, detector


def _close(presenter: AcquisitionPresenter, detector: DetectorPresenter) -> None:
    """Release the presenters a test built, without running a plan."""
    presenter.shutdown()
    detector.shutdown()


def test_the_acquisition_presenter_and_view_are_declarable() -> None:
    """Both classes pass the declaration checks Redsun applies to a session."""
    assert (
        check(AcquisitionPresenter, Layer.PRESENTER, "acq_ctrl", Qt)
        is AcquisitionPresenter
    )
    assert check(AcquisitionView, Layer.VIEW, "acq_widget", Qt) is AcquisitionView
    assert Qt.check_view(AcquisitionView, "acq_widget") is None


def test_the_acquisition_presenter_annotations_resolve_at_run_time() -> None:
    """Resolve the constructor, ``setup`` and plan annotations with ``eval_str``."""
    constructor = inspect.signature(AcquisitionPresenter, eval_str=True)
    setup = inspect.signature(AcquisitionPresenter.setup, eval_str=True)
    plan = inspect.signature(AcquisitionPresenter.live_stream, eval_str=True)

    assert list(constructor.parameters) == ["name", "devices"]
    assert list(setup.parameters) == [
        "self",
        "plan_sources",
        "callbacks",
        "paths",
    ]
    assert setup.parameters["plan_sources"].annotation == Mapping[str, HasPlans]
    assert plan.parameters["detectors"].annotation == Sequence[LiveFlyer]
    assert plan.parameters["frames"].annotation is int
    assert plan.parameters["write_forever"].annotation is bool
    assert plan.parameters["stream"].annotation is PlanAction

    hints = get_type_hints(AcquisitionPresenter.setup)
    assert hints["plan_sources"] == Mapping[str, HasPlans]
    assert hints["paths"] is SessionPathProvider


def test_the_view_constructor_matches_the_qt_session_contract() -> None:
    """Inherit the built-in page's ``(name, parent)`` constructor and placement."""
    from qtpy import QtWidgets

    parameters = list(
        inspect.signature(AcquisitionView, eval_str=True).parameters.values()
    )

    assert [(param.name, param.kind, param.annotation) for param in parameters] == [
        ("name", inspect.Parameter.POSITIONAL_OR_KEYWORD, str),
        ("parent", inspect.Parameter.POSITIONAL_OR_KEYWORD, QtWidgets.QWidget),
    ]
    assert AcquisitionView.placement == Dock("left")
    assert issubclass(AcquisitionView, BuiltinAcquisitionView)
    assert issubclass(MissingDeviceEdit, DeviceSequenceEdit)


def test_the_plan_map_offers_the_live_stream_action_in_both_forms() -> None:
    """The single action carries a frame count and a release-controlled switch."""
    presenter, detector = _acquisition()
    try:
        plan_map = presenter.plan_map()
        assert set(plan_map) == {LIVE_STREAM}
        entry = plan_map[LIVE_STREAM]
        assert entry["plan"] == presenter.live_stream

        spec = create_plan_spec(entry["plan"], {})
        parameters = {parameter.name: parameter for parameter in spec.parameters}
        assert set(parameters) == {"detectors", "frames", "write_forever", "stream"}
        assert parameters["frames"].default == 10
        assert parameters["write_forever"].default is False
        assert parameters["stream"].actions is STREAM
        assert spec.continuous
        assert presenter.plans[LIVE_STREAM].continuous

        # the finite form names its frame count, the release-controlled form
        # ignores it and ends when the action is released
        assert parameters["frames"].has_default
        assert parameters["write_forever"].has_default
    finally:
        _close(presenter, detector)


def test_the_live_stream_action_routes_to_the_detector_presenter() -> None:
    """The routed plan passes both forms to the detector presenter unchanged."""
    presenter, detector = _acquisition()
    calls: list[tuple[tuple[object, ...], int, bool, PlanAction]] = []

    def routed(
        detectors: Sequence[LiveFlyer] = (),
        frames: int = 10,
        write_forever: bool = False,
        /,
        stream: PlanAction = STREAM,
    ) -> MsgGenerator[None]:
        calls.append((tuple(detectors), frames, write_forever, stream))
        yield from ()

    detector.live_stream = routed  # type: ignore[method-assign]
    try:
        list(presenter.live_stream((), 3, False))
        list(presenter.live_stream((), 7, True, stream=STREAM))
    finally:
        _close(presenter, detector)

    assert calls == [((), 3, False, STREAM), ((), 7, True, STREAM)]


def test_a_missing_camera_keeps_a_readable_control() -> None:
    """A session without a camera unhides the selector and names the role."""
    presenter, detector = _acquisition()
    try:
        spec = presenter.plans[LIVE_STREAM]
        parameter = {item.name: item for item in spec.parameters}["detectors"]

        assert not parameter.hidden, "the parameter keeps its control"
        assert parameter.choices == [], "no device, and no invented choice"
        assert device_protocol(parameter.annotation) is LiveFlyer
        assert missing_device_notice(parameter.annotation) == CAMERA_ROLE_NOTICE
        assert is_unmatched_device(parameter)

        shown = shown_device_parameters(spec.parameters)
        assert shown[0] is spec.parameters[0]
        assert shown[0].choices == []
        assert shown[0].device_proto is LiveFlyer
        assert not shown[0].hidden
        assert shown[1] is spec.parameters[1]
    finally:
        _close(presenter, detector)


def test_a_missing_camera_does_not_raise_while_describing_the_plan() -> None:
    """Building the plan specification and resolving its values stays typed."""
    presenter, detector = _acquisition()
    try:
        spec = presenter.plans[LIVE_STREAM]
        values = resolve_arguments(
            spec,
            {"detectors": [], "frames": 10, "write_forever": False},
            {},
        )
        assert values["detectors"] == []
        assert values["frames"] == 10
        assert values["write_forever"] is False
        assert values["stream"] is STREAM
    finally:
        _close(presenter, detector)


@runtime_checkable
class _UnknownDetector(Protocol):
    """A device role the notice table does not describe yet."""

    def read(self) -> None:
        """Be a device-shaped member the predicates recognise."""
        ...


def test_the_missing_device_notice_names_an_unknown_role() -> None:
    """An unknown device annotation still gets a sentence naming its role."""
    assert missing_device_notice(_UnknownDetector) == (
        "no _UnknownDetector device declared in this session - declare one in "
        "the session profile"
    )
    assert missing_device_notice(Sequence[_UnknownDetector]) == (
        "no _UnknownDetector device declared in this session - declare one in "
        "the session profile"
    )


def test_the_missing_device_control_is_installed_exactly_once() -> None:
    """A second call adds no entry, and the entry stays first."""
    install_missing_device_control()
    install_missing_device_control()
    factories = [
        factory
        for _predicate, factory in WIDGET_FACTORY_MAP
        if factory is make_missing_device_edit
    ]
    assert factories == [make_missing_device_edit]
    assert WIDGET_FACTORY_MAP[0][1] is make_missing_device_edit


@pytest.mark.parametrize(
    ("session_type", "profile_name"),
    [(CameraSession, "camera.yaml"), (DualCameraSession, "camera-dual.yaml")],
)
def test_the_session_declares_the_acquisition_components(
    session_type: type[CameraSession], profile_name: str
) -> None:
    """Annotate and read both components, and name them in the profile."""
    hints = get_type_hints(session_type, include_extras=True)

    assert get_args(hints["acq_ctrl"])[0] is AcquisitionPresenter
    assert Layer.PRESENTER in get_args(hints["acq_ctrl"])[1:]
    assert get_args(hints["acq_widget"])[0] is AcquisitionView
    assert Layer.VIEW in get_args(hints["acq_widget"])[1:]

    profile = session_type.config
    content: Any = yaml.safe_load(profile.read_text(encoding="utf-8"))
    assert "acq_ctrl" in content["presenters"]
    assert "acq_widget" in content["views"]

    config = validate_session([profile], load(profile))
    declarations = read(session_type, config, Qt)
    assert "acq_ctrl" in declarations
    assert "acq_widget" in declarations
    assert declarations["acq_ctrl"].refusal is None
    assert declarations["acq_widget"].refusal is None


def test_the_session_declaration_order_gathers_the_detector_first() -> None:
    """The detector presenter is declared before the acquisition presenter.

    Both offer ``live_stream``; the acquisition presenter gathers the plan
    sources in declaration order, so its routed entry is the one the page runs.
    """
    hints = list(get_type_hints(CameraSession, include_extras=True))
    assert hints.index("det_ctrl") < hints.index("acq_ctrl")


def test_the_lock_signal_reaches_a_view_slot() -> None:
    """The inherited lock signal is emitted to whatever the session wires it to."""
    from redsun_aht.configurations.wiring import wire_locks

    class Holder:
        """A view stand-in that records the device names it is told about."""

        def __init__(self) -> None:
            self.names: list[frozenset[str]] = []

        def set_locked(self, names: frozenset[str]) -> None:
            self.names.append(names)

    presenter, detector = _acquisition()
    holder = Holder()
    try:
        links = list(wire_locks(presenter, holder))
        assert len(links) == 1
        signal, slot = links[0]
        assert signal is presenter.sig_locks_changed
        assert slot == holder.set_locked

        signal.connect(slot)
        presenter.sig_locks_changed.emit(frozenset({"camera1"}))
        assert holder.names == [frozenset({"camera1"})]
    finally:
        _close(presenter, detector)


def test_the_session_wires_the_page_and_the_locks() -> None:
    """The session's ``wire`` is a generator that the page and locks extend."""
    from redsun_aht.configurations.wiring import wire_acquisition, wire_locks

    assert inspect.isgeneratorfunction(CameraSession.wire)
    assert inspect.isgeneratorfunction(wire_acquisition)
    assert inspect.isgeneratorfunction(wire_locks)


def test_a_headless_session_builds_the_page_and_keeps_the_control(
    tmp_path: Path,
) -> None:
    """A real session build injects the presenters and keeps the empty camera row.

    The session resolves every annotation of an injected ``setup`` at run time
    and builds the presenters before calling it, which is what lets the
    acquisition presenter find the detector presenter among its plan sources.
    """

    class HeadlessSession(Session):
        """The acquisition and detector presenters, with no device declared."""

        config: ClassVar[dict[str, object]] = {
            "session": "aht-acquisition-headless",
            "strict": True,
            "storage": {"base_dir": str(tmp_path / "session")},
            "presenters": {"det_ctrl": {}, "acq_ctrl": {}},
        }

        det_ctrl: AsPresenter[DetectorPresenter]
        acq_ctrl: AsPresenter[AcquisitionPresenter]

    session = HeadlessSession(log_level="WARNING")
    try:
        session.build()
        assert list(session.acq_ctrl.plans) == [LIVE_STREAM]
        parameter = {
            item.name: item for item in session.acq_ctrl.plans[LIVE_STREAM].parameters
        }["detectors"]
        assert not parameter.hidden
        assert parameter.choices == []
        assert parameter.device_proto is LiveFlyer
        assert missing_device_notice(parameter.annotation) == CAMERA_ROLE_NOTICE
    finally:
        session.shutdown()
