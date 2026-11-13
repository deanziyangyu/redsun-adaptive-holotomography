"""Live-stream plan helpers over a flyer that writes image streams.

Adapted from ``redsun-planner`` (package ``sample_search``) at revision
``c0eff86``, file ``src/planner/plans.py``, Apache-2.0 licensed; copyright the
``redsun-planner`` contributors. The helpers keep the donor's message order and
docstring meaning. The donor's ``get_plan_block`` decoration and the unrelated
scan plans are not carried over.

Changed for :mod:`redsun_aht`:

- ``live_stream`` accepts an optional ``parent`` and keys its nested run
  ``capture``, so the capture can be opened while another run is open. The
  IX83 application obtains the same nesting by wrapping the donor helper with
  ``set_run_key_decorator("capture")``.
- ``teardown_live_stream`` unstages in a ``finally`` and ``live_stream``
  closes its nested run even when declaration, kickoff or teardown fails.
  A failed declaration or kickoff unstages without completing a capture that
  never started. Once kickoff succeeds, completion and collection run in a
  ``finally`` when the wrapped wait fails, the operator stops or the run is
  aborted. The RunEngine's own cleanup remains a last resort.
- Every helper refuses a detector that has no live image buffer and names the
  detector in the error.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.preprocessors import set_run_key_wrapper
from bluesky.utils import RequestAbort, RequestStop

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from bluesky.utils import MsgGenerator

    from redsun_aht.protocols import LiveFlyer

__all__ = [
    "collect_while_waiting_plan",
    "live_stream",
    "prepare_live_stream",
    "teardown_live_stream",
]

#: Run key of the nested capture, so it can be opened beside the caller's run.
_CAPTURE_RUN_KEY = "capture"


def prepare_live_stream(
    detectors: Sequence[LiveFlyer],
    trigger_info: Any,
    stream_name: str,
    *,
    collect: bool = True,
    declare: bool = True,
) -> MsgGenerator[Any]:
    """Prepare live flyers and optionally declare their event stream.

    Staging remains the caller's responsibility so one host can stage several
    device groups under its own resource policy. ``trigger_info`` is passed
    through to the Bluesky preparable protocol without importing a host type.

    Parameters
    ----------
    detectors
        The live flyers to prepare. At least one is required, and each must
        expose the live image buffer a viewer reads.
    trigger_info
        The host's trigger information, passed to each flyer's ``prepare``.
    stream_name
        The stream the detectors will write when declared. It must not be
        empty.
    collect
        Whether the declared stream is described with ``describe_collect``
        rather than ``describe``. A flyer that writes stream assets needs
        ``True``, the default.
    declare
        Whether to declare the stream after preparing. The IX83 presenter
        passes ``False`` because the nested capture declares it.
    """
    _validate_detectors(detectors)
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    for detector in detectors:
        yield from bps.prepare(detector, trigger_info, wait=True)
    if declare:
        yield from bps.declare_stream(*detectors, name=stream_name, collect=collect)


def teardown_live_stream(
    detectors: Sequence[LiveFlyer],
    stream_name: str,
    *,
    flush_period_s: float = 0.5,
    on_complete: Callable[[LiveFlyer, Any], Iterable[Any]] | None = None,
    on_flush: Callable[[LiveFlyer], Iterable[Any]] | None = None,
) -> MsgGenerator[Any]:
    """Complete live flyers while collecting written assets, then unstage.

    ``on_complete`` lets an execution host attach progress reporting without
    bringing its presenter framework into this package; it receives each
    detector and its completion status. ``on_flush`` receives each detector
    after every collection. The detectors are unstaged even when completing or
    collecting fails, so an interrupted capture leaves none of them staged.

    Parameters
    ----------
    detectors
        The live flyers to complete and unstage.
    stream_name
        The stream the flyers collect into.
    flush_period_s
        Seconds to wait for completion before collecting what was written.
    on_complete
        Optional host callback run once per detector with its status.
    on_flush
        Optional host callback run once per detector after every collection.
    """
    _validate_live_stream_inputs(detectors, stream_name, flush_period_s)
    group = uuid4().hex
    try:
        for detector in detectors:
            status = yield from bps.complete(detector, wait=False, group=group)
            if on_complete is not None:
                yield from on_complete(detector, status)
        done = False
        while not done:
            done = yield from bps.wait(
                group=group, timeout=flush_period_s, error_on_timeout=False
            )
            yield from bps.collect(*detectors, name=stream_name)
            if on_flush is not None:
                for detector in detectors:
                    yield from on_flush(detector)
    finally:
        yield from bps.unstage_all(*detectors)


def collect_while_waiting_plan(
    plan: Iterable[Any],
    detectors: Sequence[LiveFlyer],
    stream_name: str,
    *,
    flush_period_s: float = 0.5,
    on_flush: Callable[[LiveFlyer], Iterable[Any]] | None = None,
) -> MsgGenerator[Any]:
    """Collect live flyers each time a wrapped plan sleeps for a flush period.

    The elapsed value is accumulated from requested Bluesky sleep messages.
    An optional host callback can wait on device counters and update GUI
    progress after each collection.

    Parameters
    ----------
    plan
        The plan that keeps the capture open, usually a wait for an operator
        action. Its sleep messages are what schedule a collection.
    detectors
        The live flyers to collect from.
    stream_name
        The stream the flyers collect into.
    flush_period_s
        Accumulated requested sleep at which a collection happens.
    on_flush
        Optional host callback run once per detector after every collection.
    """
    _validate_live_stream_inputs(detectors, stream_name, flush_period_s)
    slept_s = 0.0

    def flush() -> Iterable[Any]:
        yield from bps.collect(*detectors, name=stream_name)
        if on_flush is not None:
            for detector in detectors:
                yield from on_flush(detector)

    def collect_after_sleep(msg: Any) -> tuple[None, Iterable[Any] | None]:
        nonlocal slept_s
        if msg.command != "sleep":
            return None, None
        slept_s += msg.args[0]
        if slept_s < flush_period_s:
            return None, None
        slept_s = 0.0
        return None, flush()

    # The Bluesky preprocessor is shipped without annotations; see the module
    # docstring for why the donor's behaviour is reproduced here.
    yield from bpp.plan_mutator(plan, collect_after_sleep)  # type: ignore[no-untyped-call]


def live_stream(
    detectors: Sequence[LiveFlyer],
    stream_name: str,
    *,
    parent: str | None = None,
    median_scan: str | None = None,
    until: Iterable[Any] | None = None,
    flush_period_s: float = 0.5,
    on_complete: Callable[[LiveFlyer, Any], Iterable[Any]] | None = None,
    on_flush: Callable[[LiveFlyer], Iterable[Any]] | None = None,
) -> MsgGenerator[str]:
    """Write one capture from already-prepared live flyers as a nested run.

    The host owns detector staging/preparation, the action UI, resource locks
    and any hardware policy. ``until`` can keep the capture open while another
    plan runs; assets are periodically collected during its sleeps. This is a
    Bluesky generator and does not create or require a RunEngine.

    Parameters
    ----------
    detectors
        The live flyers to kick off, collect from and unstage.
    stream_name
        The stream the capture declares and collects into.
    parent
        The UID of the run this capture belongs to, when it is nested in an
        open run. It is recorded in the capture's start document.
    median_scan
        The UID of the scan whose stack the capture extends, when it does.
        It is recorded in the capture's start document.
    until
        A plan that keeps the capture open. The capture ends when it returns,
        after one last completion and collection, even if the detectors were
        told to write forever.
    flush_period_s
        Seconds to wait for completion before collecting what was written.
    on_complete
        Optional host callback run once per detector with its status.
    on_flush
        Optional host callback run once per detector after every collection.

    Returns
    -------
    str
        The UID of the nested capture run.

    Notes
    -----
    The capture opens under the run key ``"capture"`` so that an enclosing run
    can stay open under the default key. The normal message order is the donor's:
    open the run, declare the stream, kick off, wait on ``until``, complete,
    collect and unstage, then close the run. Failed declaration or kickoff
    unstages the detectors; all paths after opening close the nested run.
    """
    _validate_live_stream_inputs(detectors, stream_name, flush_period_s)
    if parent is not None and (not isinstance(parent, str) or not parent.strip()):
        raise ValueError("parent must be a nonempty run UID when supplied")
    if median_scan is not None and (
        not isinstance(median_scan, str) or not median_scan.strip()
    ):
        raise ValueError("median_scan must be a nonempty run UID when supplied")

    def capture() -> MsgGenerator[str]:
        uid: str = yield from bps.open_run(
            md={"purpose": "capture", "parent": parent, "median_scan": median_scan}
        )
        kicked_off = False
        exit_status = "success"
        try:
            yield from bps.declare_stream(*detectors, name=stream_name, collect=True)
            yield from bps.kickoff_all(*detectors, wait=True)
            kicked_off = True
            if until is not None:
                yield from collect_while_waiting_plan(
                    until,
                    detectors,
                    stream_name,
                    flush_period_s=flush_period_s,
                    on_flush=on_flush,
                )
        except RequestAbort:
            exit_status = "abort"
            raise
        except RequestStop:
            raise
        except BaseException:
            exit_status = "fail"
            raise
        finally:
            try:
                if kicked_off:
                    yield from teardown_live_stream(
                        detectors,
                        stream_name,
                        flush_period_s=flush_period_s,
                        on_complete=on_complete,
                        on_flush=on_flush,
                    )
                else:
                    yield from bps.unstage_all(*detectors)
            except RequestAbort:
                exit_status = "abort"
                raise
            except RequestStop:
                raise
            except BaseException:
                exit_status = "fail"
                raise
            finally:
                yield from bps.close_run(exit_status=exit_status)
        return uid

    return (yield from _with_capture_run_key(capture()))


def _with_capture_run_key(plan: MsgGenerator[str]) -> MsgGenerator[str]:
    """Re-key *plan*'s messages so its run nests beside the caller's run."""
    # The Bluesky preprocessor is shipped without annotations; see the module
    # docstring for why the donor's behaviour is reproduced here.
    keyed = set_run_key_wrapper(plan, _CAPTURE_RUN_KEY)  # type: ignore[no-untyped-call]
    return cast("MsgGenerator[str]", keyed)


def _validate_detectors(detectors: Sequence[LiveFlyer]) -> None:
    """Refuse a capture with no detector or with one that has no live buffer."""
    if not detectors:
        raise ValueError("at least one live detector is required")
    for detector in detectors:
        if getattr(detector, "buffer", None) is None:
            name = getattr(detector, "name", "<unnamed detector>")
            raise ValueError(
                f"live detector {name!r} has no buffer signal; a live stream "
                "needs the image buffer a viewer reads"
            )


def _validate_live_stream_inputs(
    detectors: Sequence[LiveFlyer], stream_name: str, flush_period_s: float
) -> None:
    """Validate shared live-stream inputs before emitting Bluesky messages."""
    _validate_detectors(detectors)
    if not stream_name:
        raise ValueError("stream_name must not be empty")
    if (
        isinstance(flush_period_s, bool)
        or not isinstance(flush_period_s, (int, float))
        or not math.isfinite(float(flush_period_s))
        or flush_period_s <= 0
    ):
        raise ValueError("flush_period_s must be finite and positive")
