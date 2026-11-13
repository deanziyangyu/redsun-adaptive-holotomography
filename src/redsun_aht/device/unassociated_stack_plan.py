"""Headless Bluesky plan for experimental unassociated stack evidence."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp

from redsun_aht.device.unassociated_stack import acquire_unassociated_stack

if TYPE_CHECKING:
    from collections.abc import Generator

    from redsun_aht.device.pva_capture_evidence import PvaCaptureEvidence
    from redsun_aht.device.unassociated_stack import ReservableStackClient
    from redsun_aht.device.unified_mcu import AngularStack


class _Result:
    name = "unassociated_stack"

    def __init__(self, controller: str, capture: str) -> None:
        self.values = {
            "controller_report_json": controller,
            "camera_capture_json": capture,
        }

    def read(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        return {
            key: {"value": value, "timestamp": now}
            for key, value in self.values.items()
        }

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "source": "ARH:PVA:unassociated-evidence",
                "dtype": "string",
                "shape": [],
            }
            for key in self.values
        }


async def _failure_snapshot(
    client: ReservableStackClient, camera: PvaCaptureEvidence, stack: AngularStack
) -> dict[str, Any]:
    """Read best-effort reports; their absence and stop state remain uncertain."""
    evidence: dict[str, Any] = {"controller": None, "camera": None, "read_errors": {}}
    for name, read in (
        ("controller", lambda: getattr(client, "read_progress")(stack)),  # noqa: B009
        ("camera", camera.snapshot),
    ):
        try:
            evidence[name] = asdict(await asyncio.wait_for(read(), timeout=1))
        except Exception as error:
            evidence["read_errors"][name] = f"{type(error).__name__}: {error}"
    return evidence


def unassociated_stack_plan(
    client: ReservableStackClient,
    camera: PvaCaptureEvidence,
    stack: AngularStack,
    uri: str,
    *,
    resources_exclusive: bool,
    external_trigger_configured: bool,
    illumination_ready: bool,
    timeout_s: float,
) -> Generator[Any, Any, None]:
    """Run one externally configured stack without inventing frame associations."""

    @bpp.run_decorator(
        md={
            "stack_definition": asdict(stack),
            "store_uri": uri,
            "association": "unassociated",
        }
    )  # type: ignore[untyped-decorator]
    def run() -> Generator[Any, Any, None]:
        tasks = yield from bps.wait_for(
            [
                lambda: acquire_unassociated_stack(
                    client,
                    camera,
                    stack,
                    uri,
                    resources_exclusive=resources_exclusive,
                    external_trigger_configured=external_trigger_configured,
                    illumination_ready=illumination_ready,
                    timeout_s=timeout_s,
                )
            ]
        )
        try:
            result = tasks[0].result()
        except Exception as error:
            snapshots = yield from bps.wait_for(
                [lambda: _failure_snapshot(client, camera, stack)]
            )
            evidence = snapshots[0].result()
            evidence["error"] = f"{type(error).__name__}: {error}"
            evidence["quality"] = "unassociated_stop_unverified"
            source = _Result("", "")
            source.values = {"failure_evidence_json": json.dumps(evidence)}
            yield from bps.create(name="unassociated_stack_failure")
            yield from bps.read(source)
            yield from bps.save()
            raise
        source = _Result(
            json.dumps(asdict(result.controller)), json.dumps(asdict(result.capture))
        )
        yield from bps.create(name="unassociated_stack_evidence")
        yield from bps.read(source)
        yield from bps.save()

    yield from run()
