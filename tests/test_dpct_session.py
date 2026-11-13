"""Tests for the DPCT session: declaration, plan map, notice and registry.

Nothing here builds the session, starts a service, opens a serial port or
constructs a Qt application. The session class is read the way Redsun reads it
before it builds anything, the profile and the manifest are resolved with
``importlib`` so a rename breaks a test rather than startup, and the presenter
is exercised with fakes whose messages never leave a ``RunEngine``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_type_hints

import pytest
import yaml
from bluesky import RunEngine
from bluesky import plan_stubs as bps

import redsun_aht
import redsun_aht.configurations as configurations
from redsun_aht.contracts import (
    FrameIdentity,
    PartialRun,
    PlanAbortReason,
    PlanOutcome,
    PlanStatus,
    RunIdentity,
)
from redsun_aht.device.arduino.illumination import ArduinoIllumination
from redsun_aht.presenter import dpct as dpct_presenter
from redsun_aht.presenter.acquisition import CAMERA_ROLE_NOTICE
from redsun_aht.presenter.dpct import (
    DPCT_PLAN,
    ILLUMINATION_ROLE_NOTICE,
    DpctPresenter,
    describe_outcome,
    new_run_id,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from bluesky.protocols import Reading, Status
    from bluesky.utils import MsgGenerator


def _package_root() -> Path:
    """Return the installed package directory."""
    assert redsun_aht.__file__ is not None
    return Path(redsun_aht.__file__).resolve().parent


PACKAGE = _package_root()
MANIFEST = PACKAGE / "redsun.yaml"
PROFILES = PACKAGE / "profiles"
DPCT_PROFILE = PROFILES / "dpct.yaml"


def _manifest() -> dict[str, Any]:
    """Return the plugin manifest as a mapping."""
    data: Any = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _profile() -> dict[str, Any]:
    """Return the packaged DPCT profile as a mapping."""
    data: Any = yaml.safe_load(DPCT_PROFILE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _resolve(path: str) -> Any:
    """Import the ``module:attribute`` path *path* names."""
    module_name, separator, attribute = path.partition(":")
    assert separator == ":" and module_name and attribute, path
    return getattr(importlib.import_module(module_name), attribute)


@dataclass
class _CompletedStatus:
    """A Bluesky status that is already complete when it is returned."""

    def add_callback(self, callback: Callable[[Status], None]) -> None:
        callback(self)

    def exception(self, timeout: float | None = 0.0) -> BaseException | None:
        return None

    @property
    def done(self) -> bool:
        return True

    @property
    def success(self) -> bool:
        return True


class _FrameCounter:
    """A readable integer counter of delivered frames."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.value = 0

    def read(self) -> dict[str, Reading[Any]]:
        return {self.name: {"value": self.value, "timestamp": 0.0}}


class _Camera:
    """An armable camera the presenter expects one frame from per pattern."""

    def __init__(self, name: str = "camera1") -> None:
        self.name = name
        self.counter = _FrameCounter(f"{name}-frame-count")
        self.prepared: list[Any] = []

    @property
    def frame_count(self) -> _FrameCounter:
        """Return the counter a DPCT plan polls."""
        return self.counter

    def stage(self) -> Status:
        return _CompletedStatus()

    def unstage(self) -> Status:
        return _CompletedStatus()

    def prepare(self, value: Any) -> Status:
        self.prepared.append(value)
        return _CompletedStatus()


class _Stage:
    """A stage the presenter may hand to the plan; the faked plan never reads it."""

    def __init__(self, name: str = "stage1") -> None:
        self.name = name
        self.axes: Any = {}

    async def read_position(self) -> Any:
        raise NotImplementedError


class _Led:
    """An Arduino LED device, reduced to the surface the presenter wraps."""

    pattern: Any = None
    all_off: Any = None
    open_shutter: Any = None
    last_command: Any = None
    last_outcome: Any = None
    last_detail: Any = None


def _presenter(
    *, camera: bool = True, stage: bool = True, illumination: bool = True
) -> DpctPresenter:
    """Return a presenter over whichever devices are asked for."""
    return DpctPresenter(
        "dpct_ctrl",
        detectors={"camera1": _Camera()} if camera else {},
        stage={"stage1": _Stage()} if stage else {},
        illumination={"led1": _Led()} if illumination else {},
    )


def _run(engine: RunEngine, plan: MsgGenerator[PlanOutcome]) -> list[PlanOutcome]:
    """Run *plan* and collect the outcome it returned."""
    returned: list[PlanOutcome] = []

    def wrapper() -> Any:
        returned.append((yield from plan))

    engine(wrapper())
    return returned


def test_the_session_declaration_resolves() -> None:
    """Resolve every declared class the way Redsun checks it before building."""
    from redsun import Layer
    from redsun.session._declarations import check

    from redsun_aht.configurations.dpct import DpctSession
    from redsun_aht.device.arduino.device import ArduinoLedDevice
    from redsun_aht.device.mmcore import MMStage
    from redsun_aht.presenter.dpct import DpctPresenter

    assert DPCT_PROFILE.is_file()
    assert DpctSession.config == DPCT_PROFILE

    assert check(MMStage, Layer.DEVICE, "stage1") is MMStage
    assert check(ArduinoLedDevice, Layer.DEVICE, "led1") is ArduinoLedDevice
    assert check(DpctPresenter, Layer.PRESENTER, "dpct_ctrl") is DpctPresenter


def test_the_session_annotations_carry_their_layers() -> None:
    """Declare the stage and LED as devices and the DPCT presenter as a presenter."""
    from redsun import Layer

    from redsun_aht.configurations.dpct import DpctSession

    hints = get_type_hints(DpctSession, include_extras=True)

    for name in ("stage1", "led1"):
        assert Layer.DEVICE in get_args(hints[name])[1:], name
    assert Layer.PRESENTER in get_args(hints["dpct_ctrl"])[1:]


def test_the_profile_declares_every_annotated_component() -> None:
    """Read the session's declarations from its packaged profile without building it."""
    from redsun._config import load, validate_session
    from redsun.qt import Qt
    from redsun.session._declarations import read

    from redsun_aht.configurations.dpct import DpctSession

    config = validate_session([DPCT_PROFILE], load(DPCT_PROFILE))
    declarations = read(DpctSession, config, Qt)

    assert set(declarations) == {
        "acq_ctrl",
        "acq_widget",
        "camera1",
        "det_ctrl",
        "det_widget",
        "dpct_ctrl",
        "img_widget",
        "led1",
        "stage1",
    }
    for name, declaration in declarations.items():
        assert declaration.refusal is None, f"{name}: {declaration.refusal}"


def test_the_plan_map_names_the_dpct_plan() -> None:
    """Offer one parameterless plan, named as the run's plan name."""
    presenter = _presenter()

    plans = presenter.plan_map()

    assert presenter.missing_notice() is None
    assert set(plans) == {DPCT_PLAN}
    assert plans[DPCT_PLAN]["plan"] == presenter.dpct


def test_the_plan_signature_is_describable() -> None:
    """Describe the plan the way the acquisition page builds its controls."""
    from redsun.presenter.plan_spec import create_plan_spec

    spec = create_plan_spec(_presenter().dpct, {})

    assert spec.name == DPCT_PLAN
    assert spec.parameters == []


def test_a_missing_illumination_device_is_reported_and_offers_no_plan() -> None:
    """Build without illumination, name the role and offer nothing to run."""
    presenter = _presenter(illumination=False)

    assert presenter.missing_notice() == ILLUMINATION_ROLE_NOTICE
    assert presenter.plan_map() == {}


def test_a_missing_camera_is_reported_and_offers_no_plan() -> None:
    """Build without a camera and name the camera role the page is missing."""
    presenter = _presenter(camera=False)

    assert presenter.missing_notice() == CAMERA_ROLE_NOTICE
    assert presenter.plan_map() == {}


def test_a_run_without_devices_returns_an_aborted_outcome() -> None:
    """Report a missing device as an outcome instead of raising from the plan."""
    presenter = _presenter(illumination=False)
    engine = RunEngine({})

    (outcome,) = _run(engine, presenter.dpct())

    assert outcome.status is PlanStatus.ABORTED
    assert outcome.reason is PlanAbortReason.DEVICE_FAULT
    assert outcome.detail == ILLUMINATION_ROLE_NOTICE
    assert describe_outcome(outcome).startswith(f"dpct run {outcome.run.run_id}")


def test_the_plan_is_bound_to_the_session_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call the software plan with the session devices and wrapped LED device."""
    camera = _Camera()
    stage = _Stage()
    led = _Led()
    run = RunIdentity(
        run_id="run-a", plan_name=DPCT_PLAN, purpose="four-pattern DPCT", scan_id=0
    )
    outcome = PlanOutcome(
        run=run, status=PlanStatus.COMPLETE, detail="4 patterns, 4 frames"
    )
    recorded: dict[str, Any] = {}

    def fake(
        detectors: Any,
        stage_arg: Any,
        illumination: Any,
        *,
        run_id: str,
    ) -> MsgGenerator[PlanOutcome]:
        recorded.update(
            detectors=detectors,
            stage=stage_arg,
            illumination=illumination,
            run_id=run_id,
        )
        yield from bps.null()
        return outcome

    monkeypatch.setattr(dpct_presenter, "software_dpct_plan", fake)
    presenter = DpctPresenter(
        "dpct_ctrl",
        detectors={"camera1": camera},
        stage={"stage1": stage},
        illumination={"led1": led},
    )
    announced: list[PlanOutcome] = []
    presenter.sig_outcome.connect(announced.append)
    engine = RunEngine({})

    returned = _run(engine, presenter.dpct())

    assert returned == [outcome]
    assert announced == [outcome]
    assert recorded["detectors"] == (camera,)
    assert recorded["stage"] is stage
    assert isinstance(recorded["illumination"], ArduinoIllumination)
    run_id = recorded["run_id"]
    assert isinstance(run_id, str)
    assert run_id.startswith(f"{DPCT_PLAN}-")
    assert not any(character.isspace() for character in run_id)


def test_a_stage_is_optional() -> None:
    """Hand the plan no stage when the session declares none."""
    presenter = _presenter(stage=False)

    assert presenter.missing_notice() is None
    assert set(presenter.plan_map()) == {DPCT_PLAN}


def test_the_run_id_is_a_fresh_whitespace_free_token() -> None:
    """Mint one store key per run, from a UTC stamp and a random token."""
    run_id = new_run_id()

    assert run_id.startswith(f"{DPCT_PLAN}-")
    assert len(run_id.encode("utf-8")) <= 96
    assert not any(character.isspace() for character in run_id)
    assert new_run_id() != run_id


def test_a_complete_outcome_is_reported_as_complete() -> None:
    """Say the run completed and repeat the plan's own summary."""
    run = RunIdentity(
        run_id="run-a", plan_name=DPCT_PLAN, purpose="four-pattern DPCT", scan_id=0
    )
    outcome = PlanOutcome(
        run=run, status=PlanStatus.COMPLETE, detail="4 patterns, 4 frames"
    )

    assert describe_outcome(outcome) == "dpct run run-a complete: 4 patterns, 4 frames"


def test_a_partial_outcome_names_its_missing_ordinals_and_patterns() -> None:
    """Name what arrived beside the ordinals and patterns that did not."""
    run = RunIdentity(
        run_id="run-b", plan_name=DPCT_PLAN, purpose="four-pattern DPCT", scan_id=0
    )
    requested = tuple(
        FrameIdentity(
            run=run,
            detector_id="camera1",
            sequence=ordinal,
            plane_ordinal=ordinal,
            pattern_id=ordinal + 1,
            scan_index=0,
        )
        for ordinal in range(4)
    )
    outcome = PlanOutcome(
        run=run,
        status=PlanStatus.ABORTED,
        reason=PlanAbortReason.CAMERA_TIMEOUT,
        partial=PartialRun(requested=requested, produced=requested[:2]),
        detail="detector 'camera1' delivered no frame for pattern 3",
    )

    report = describe_outcome(outcome)

    assert report.startswith("dpct run run-b aborted (camera_timeout): ")
    assert "2 of 4 frames produced" in report
    assert "missing ordinals 2, 3" in report
    assert "patterns 3, 4" in report
    assert "delivered no frame for pattern 3" in report


def test_the_factory_returns_the_dpct_session() -> None:
    """Choose the packaged profile, or the caller's, without building anything."""
    from redsun_aht.configurations.dpct import DpctSession, build_dpct_session

    packaged = build_dpct_session()

    assert isinstance(packaged, DpctSession)
    assert packaged.config == DPCT_PROFILE
    assert isinstance(build_dpct_session(DPCT_PROFILE), DpctSession)


def test_the_cli_registry_lists_the_dpct_session() -> None:
    """Expose the sessions and services required by the DPCT workflow."""
    sessions, services = configurations.registry()

    assert {"camera", "dpct"} <= sessions.keys()
    assert {"mmcore-camera", "mmcore-stage", "arduino-led"} <= services.keys()
    assert (sessions["dpct"].__doc__ or "").strip()
    assert "DpctSession" in dir(configurations)
    assert configurations.DpctSession.__name__ == "DpctSession"


def test_every_manifest_and_profile_plugin_id_resolves() -> None:
    """Resolve every manifest path and every service id the profile names."""
    manifest = _manifest()
    profile = _profile()

    for section in ("devices", "presenters", "views"):
        for name, path in manifest[section].items():
            assert isinstance(_resolve(path), type), f"{section} {name}: {path}"
    for name, entry in manifest["services"].items():
        module = importlib.import_module(entry["module"])
        assert callable(getattr(module, "main", None)), f"service {name}"

    assert _resolve(manifest["presenters"]["dpct"]).__name__ == "DpctPresenter"

    assert profile["schema_version"] == 1.0
    assert profile["session"]
    assert profile["services"]["transport"] == "pv-access"
    named = {
        key: value for key, value in profile["services"].items() if key != "transport"
    }
    assert set(named) == {"camera1", "stage1", "led1"}
    for name, service in named.items():
        assert service["plugin_name"] == "redsun-aht", name
        assert service["plugin_id"] in manifest["services"], name
        assert service["prefix"].endswith(":"), name
    led = named["led1"]
    assert led["args"]["prefix"] == led["prefix"]
    assert led["args"]["port"]
    assert {"det_ctrl", "acq_ctrl", "dpct_ctrl"} <= set(profile["presenters"])
    assert {"det_widget", "acq_widget", "img_widget"} <= set(profile["views"])


def test_the_packaged_serial_port_is_marked_as_a_placeholder() -> None:
    """Say in the profile that the port must be replaced before hardware use."""
    text = DPCT_PROFILE.read_text(encoding="utf-8")

    assert "PLACEHOLDER" in text
    assert "COM19" in text
