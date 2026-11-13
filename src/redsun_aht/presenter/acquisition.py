"""The acquisition page's plan gathering, with a control for every device role.

Ported from ``redsun_searching_is_imaging/presenter/acquisition.py`` at donor
revision ``3a3bf06`` (``redsun-searching-is-imaging``), Apache-2.0 licensed;
copyright the Redsun contributors. The plan gathering, the rule that a device
parameter no session device matches keeps its control, and the sentence naming
the missing role are the donor's.

Changed for :mod:`redsun_aht`:

- The class is :class:`AcquisitionPresenter`, the name this bundle's session
  declares. It also offers the ``live_stream`` action itself, through
  :meth:`AcquisitionPresenter.live_stream`, which routes to the session's
  :class:`~redsun_aht.presenter.detector.DetectorPresenter`. The donor's
  searching bundle let the detector's plan map reach the page only through its
  presenter; here the acquisition presenter is the page's single entry, so the
  finite and the release-controlled forms are named together and the routing is
  explicit.
- ``setup`` finds the detector presenter among the plan sources, so the plan
  can reach the detector that owns the buffer monitor, the store and the
  ``stream`` action without adding a parameter the framework's own ``setup``
  does not take. The base presenter supplies the ``Deferrals`` the detector
  presenter takes in its own ``setup``.
- The role table names this bundle's device annotations. A role it does not
  know still reads as a role: the annotation's own name stands in, as in the
  donor.
- ``sig_locks_changed`` is inherited from :class:`redsun.presenter.
  AcquisitionPresenter` and is the signal a session wires to a view's
  ``set_locked``; a view disables the controls of the devices a running plan
  holds. The donor wired the same signal from its acquisition presenter.
"""

from __future__ import annotations

# These names are imported at run time on purpose: a Redsun session resolves
# the constructor and ``setup`` annotations with ``inspect.signature(...,
# eval_str=True)`` against this module's globals, and a plan map is described
# by resolving the plan's annotations the same way. Moving any name a
# constructor, ``setup`` or plan annotation uses under ``TYPE_CHECKING`` makes
# the presenter undeclarable.
from collections.abc import Mapping, Sequence  # noqa: TC003
from dataclasses import replace
from typing import Any, Final, cast, get_args

from bluesky.utils import MsgGenerator  # noqa: TC002
from redsun.engine.actions import PlanAction, continuous
from redsun.path_provider import SessionPathProvider  # noqa: TC002
from redsun.presenter import AcquisitionPresenter as _AcquisitionPresenter
from redsun.presenter.plan_spec import ParamDescription  # noqa: TC002
from redsun.presenter.utils import (
    device_class,
    isdevice,
    isdevicesequence,
    isdeviceset,
)
from redsun.registry import CallbackType, HasPlans, PlanEntry

from redsun_aht.presenter.detector import (
    LIVE_STREAM,
    STREAM,
    DetectorPresenter,
)
from redsun_aht.protocols import LiveFlyer  # noqa: TC001

__all__ = [
    "CAMERA_ROLE_NOTICE",
    "STAGE_ROLE_NOTICE",
    "AcquisitionPresenter",
    "device_protocol",
    "missing_device_notice",
    "shown_device_parameters",
]

CAMERA_ROLE_NOTICE: Final = (
    "no camera declared in this session - declare an mmcore camera device in the "
    "session profile"
)
"""What a camera control says when the session declares no camera."""

STAGE_ROLE_NOTICE: Final = (
    "no stage declared in this session - declare an mmcore stage device in the "
    "session profile"
)
"""What a stage control says when the session declares no stage."""

_ROLE_NOTICES: Final[Mapping[str, str]] = {
    # the camera role, under every protocol and class the plans ask for it with
    "DetectorDevice": CAMERA_ROLE_NOTICE,
    "LiveFlyer": CAMERA_ROLE_NOTICE,
    "MMCamera": CAMERA_ROLE_NOTICE,
    # the stage role
    "MMStage": STAGE_ROLE_NOTICE,
    "MotorProtocol": STAGE_ROLE_NOTICE,
    "XYMotor": STAGE_ROLE_NOTICE,
}
"""The sentence for each protocol and class name a device annotation may carry.

A name is what a session device is matched against, not a role of its own:
several protocols and classes stand for the camera role and several for the
stage role.
"""


def device_protocol(annotation: Any) -> type[Any] | None:
    """Return the class a device annotation matches session devices against.

    A device sequence or set is matched by its only argument and a plain device
    annotation by itself, as `plan_spec._fields_from_annotation` matches them.
    Returns ``None`` for an annotation naming no device at all.

    Parameters
    ----------
    annotation : Any
        The resolved annotation of a plan parameter.

    Returns
    -------
    type[Any] | None
        The class or runtime-checkable protocol the session's devices are
        checked against, or ``None``.
    """
    if isdevicesequence(annotation) or isdeviceset(annotation):
        return cast("type[Any]", device_class(get_args(annotation)[0]))
    if isdevice(annotation):
        return cast("type[Any]", device_class(annotation))
    return None


def missing_device_notice(annotation: Any) -> str:
    """Return the sentence naming the device role *annotation* asks for.

    A role no entry describes still reads as a role: the annotation's own name
    stands in, so a new device protocol never leaves the operator with a control
    that says nothing.

    Parameters
    ----------
    annotation : Any
        The resolved annotation of a device parameter.

    Returns
    -------
    str
        The readable sentence naming the missing role.
    """
    proto = device_protocol(annotation)
    if proto is None:
        return f"no device declared in this session for {annotation!r}"
    for name in _names_of(proto):
        notice = _ROLE_NOTICES.get(name)
        if notice is not None:
            return notice
    return (
        f"no {proto.__name__} device declared in this session - declare one in "
        "the session profile"
    )


def shown_device_parameters(
    parameters: Sequence[ParamDescription],
) -> list[ParamDescription]:
    """Return *parameters* with a control kept for every device role the session lacks.

    A device parameter with no candidate device keeps its place on the page: it
    is described with no choice at all and is no longer hidden, so the page's
    *Devices* group holds its control, which the view labels with
    :func:`missing_device_notice`. Nothing else about the parameter changes, and
    a parameter with candidates is returned untouched.

    Parameters
    ----------
    parameters : Sequence[ParamDescription]
        The parameters a plan's specification was built from.

    Returns
    -------
    list[ParamDescription]
        The same parameters, with unmatched device parameters unhidden.
    """
    return [_shown(parameter) for parameter in parameters]


def _shown(parameter: ParamDescription) -> ParamDescription:
    """Return *parameter* with its control kept when no session device matches it."""
    proto = device_protocol(parameter.annotation)
    if proto is None or not parameter.hidden:
        return parameter
    return replace(parameter, choices=[], device_proto=proto, hidden=False)


def _names_of(proto: type[Any]) -> tuple[str, ...]:
    """Return *proto*'s name and its bases', most specific first."""
    bases: tuple[type[Any], ...] = getattr(proto, "__mro__", (proto,))
    return tuple(base.__name__ for base in bases)


class AcquisitionPresenter(_AcquisitionPresenter):
    """Gather the session's plans and offer the live-stream action.

    The framework's ``setup`` gathers the plan map of every component and hides
    a device parameter no session device matches, dropping the operator's
    control from a plan that stays listed. This presenter keeps such a
    parameter, empty and labelled with the role the session does not declare,
    and adds the live-stream action that routes to the session's
    :class:`~redsun_aht.presenter.detector.DetectorPresenter`.

    The action has one finite form and one release-controlled form, as the
    detector presenter's plan does: a frame count writes that many frames and
    stops, while ``write_forever`` keeps the capture open until the ``stream``
    action is released. The action button on the page is the detector
    presenter's ``stream`` action.

    ``sig_locks_changed`` is inherited from the Redsun presenter: a session
    wires it to a view's ``set_locked`` so the controls of the devices a
    running plan holds are disabled until it ends.

    ``setup`` is the method the session's injection fills, so what it is
    annotated with is imported at run time rather than under ``TYPE_CHECKING``:
    the graph evaluates every annotation of an injected method.
    """

    _detector: DetectorPresenter | None
    """The detector presenter whose live-stream plan this one routes to."""

    def __post_init__(self) -> None:
        """Build the engine and deferrals of the base, before a detector is given."""
        super().__post_init__()
        self._detector = None

    def setup(
        self,
        plan_sources: Mapping[str, HasPlans],
        callbacks: Mapping[str, CallbackType],
        paths: SessionPathProvider,
    ) -> None:
        """Gather the plans the framework can describe, then keep their device controls.

        The detector presenter is found among the plan sources before they are
        gathered, because this presenter is one of them: its own plan map is
        added to the sources when a caller supplies only the other presenters,
        so the ``live_stream`` action is on the page in every session shape.
        The framework's own bookkeeping is left as it is - the plans, the
        entries, the callbacks and the action owners - so only the
        specification each page is built from changes, and it changes in place.

        Parameters
        ----------
        plan_sources : Mapping[str, HasPlans]
            The components offering plans, by name; one of them is the
            detector presenter this page routes its live stream to.
        callbacks : Mapping[str, CallbackType]
            The document callbacks a user may attach to a run.
        paths : SessionPathProvider
            The provider naming the directory runs write under.

        Raises
        ------
        ValueError
            If no plan source is a detector presenter, so there is nothing to
            route the live stream to.
        """
        detector = next(
            (
                source
                for source in plan_sources.values()
                if isinstance(source, DetectorPresenter)
            ),
            None,
        )
        if detector is None:
            raise ValueError(
                "the acquisition presenter needs the detector presenter; "
                "declare a DetectorPresenter in the session"
            )
        self._detector = detector
        sources = dict(plan_sources)
        sources.setdefault(self.name, self)
        super().setup(sources, callbacks, paths)
        for spec in self.plans.values():
            spec.parameters[:] = shown_device_parameters(spec.parameters)

    def plan_map(self) -> Mapping[str, PlanEntry]:
        """Return the live-stream action this presenter adds to the page.

        Returns
        -------
        Mapping[str, PlanEntry]
            One entry, ``"live_stream"``, whose plan is
            :meth:`AcquisitionPresenter.live_stream`.
        """
        return {LIVE_STREAM: PlanEntry(plan=self.live_stream)}

    @continuous
    def live_stream(
        self,
        detectors: Sequence[LiveFlyer] = (),
        frames: int = 10,
        write_forever: bool = False,
        /,
        stream: PlanAction = STREAM,
    ) -> MsgGenerator[None]:
        """Record the selected detectors through the detector presenter's plan.

        Every acquired frame is forwarded to the detector presenter's
        ``sig_new_data`` through the monitor on each detector's buffer. The
        ``stream`` action writes the acquired frames to a store for ``frames``
        frames, as a run nested in this one whose run key is ``capture``. With
        ``write_forever`` the capture stays open until the action is released.
        Live visualisation continues meanwhile.

        Parameters
        ----------
        detectors : Sequence[LiveFlyer]
            The detectors to collect from, ticked from the session's streaming
            devices. Leaving it empty streams every device the session declares
            that satisfies :class:`~redsun_aht.protocols.LiveFlyer`.
        frames : int
            The number of frames to stream to disk in the finite form.
        write_forever : bool
            Stream until the ``stream`` action is released, ignoring ``frames``.
        stream : PlanAction
            The action that starts and stops writing to disk; it is the
            detector presenter's action, so the page button drives the same
            manager the plan waits on.

        Raises
        ------
        ValueError
            If no detector presenter was given in ``setup``, so there is
            nothing to route the plan to.
        """
        detector = self._detector
        if detector is None:
            raise ValueError(
                "the acquisition presenter needs the detector presenter; "
                "declare a DetectorPresenter in the session"
            )
        yield from detector.live_stream(detectors, frames, write_forever, stream)
