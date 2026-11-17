import unittest
from types import SimpleNamespace as NS
import json
from pathlib import Path
from uuid import uuid4
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from bluesky import RunEngine
from ophyd_async.core import soft_signal_rw
from redsun.aio import run_coro

from redsun_aht.presenter.arh import ArhPresenter
from redsun_aht.storage.writer import manifest_path, timing_sidecar_path


class TestDueStackPresenter(unittest.TestCase):
    def test_recovery_plans_work_without_camera_binding(self):
        enabled = AsyncMock(return_value=False)
        status_request = soft_signal_rw(int, 0, name="status_request")
        abort = soft_signal_rw(int, 0, name="abort")
        blank = soft_signal_rw(bool, False, name="blank")
        mcu = NS(
            zdt_stack_enabled=NS(get_value=enabled),
            zdt_status_request=status_request,
            zdt_abort=abort,
            blank=blank,
            zdt_status=NS(get_value=AsyncMock(
                return_value='{"run_id":7,"state":"cancelled","edges":0}',
            )),
        )
        plans = ArhPresenter("arh").plan_map()
        engine = RunEngine({}, context_managers=[])
        for name in ("due_z_stack_status", "due_z_stack_abort"):
            with self.subTest(plan=name), self.assertRaisesRegex(RuntimeError, "has not enabled"):
                engine(plans[name]["plan"](mcu, 7))
        self.assertEqual(run_coro(status_request.get_value()), 0)
        self.assertEqual(run_coro(abort.get_value()), 0)
        self.assertFalse(run_coro(blank.get_value()))

        enabled.return_value = True
        for name in ("due_z_stack_status", "due_z_stack_abort"):
            engine(plans[name]["plan"](mcu, 7))
        self.assertEqual(run_coro(status_request.get_value()), 7)
        self.assertEqual(run_coro(abort.get_value()), 7)
        self.assertTrue(run_coro(blank.get_value()))

    def test_fault_status_is_visible_in_plan_error(self):
        cause = RuntimeError("Due stack reported fault_unverified")
        mcu = NS(zdt_stack_enabled=NS(get_value=AsyncMock(return_value=True)))
        presenter = ArhPresenter("arh", due_capture_factory=Mock(), due_store_uri=Mock(return_value="unused"))
        engine = RunEngine({}, context_managers=[])
        for fault in (cause, ExceptionGroup("tasks failed", [cause])):
            fault.add_note("Check due_z_stack_status for the run's outcome.")
            with self.subTest(error=type(fault).__name__):
                with patch("redsun_aht.presenter.arh.capture_due_stack", side_effect=fault):
                    with self.assertRaisesRegex(RuntimeError, "fault_unverified.*due_z_stack_status"):
                        engine(presenter.due_z_stack(mcu, 7, 0, 1, 2, 0, 2, True, True))

    def test_disabled_stack_leaves_camera_and_storage_unopened(self):
        camera_factory, store_uri = Mock(), Mock()
        enabled = AsyncMock(return_value=False)
        mcu = NS(zdt_stack_enabled=NS(get_value=enabled))
        presenter = ArhPresenter(
            "arh", due_capture_factory=camera_factory, due_store_uri=store_uri,
        )
        engine = RunEngine({}, context_managers=[])
        with self.assertRaisesRegex(RuntimeError, "has not enabled ZDT-X stacks"):
            engine(presenter.due_z_stack(mcu, 7, 0, 1, 2, 0, 2, True, True))
        enabled.assert_awaited_once_with(cached=False)
        camera_factory.assert_not_called()
        store_uri.assert_not_called()

    def test_out_of_bounds_stack_rejected_before_camera_or_store(self):
        camera_factory = Mock()
        store_uri = Mock()
        presenter = ArhPresenter(
            "arh", due_capture_factory=camera_factory, due_store_uri=store_uri,
        )
        plan = presenter.due_z_stack(NS(), 7, 0, 2, 3, 0, 3, True, True)
        with self.assertRaisesRegex(ValueError, "stack targets exceed bounds"):
            next(plan)
        camera_factory.assert_not_called()
        store_uri.assert_not_called()

    def test_run_stores_each_frame_and_reports_completion(self):
        frames = [NS(sequence=i, receipt_ns=i * 100, data=np.full((2, 2), i)) for i in (1, 2)]
        camera = NS(
            arm=AsyncMock(return_value=0), confirm_ready=AsyncMock(),
            collect=AsyncMock(return_value=frames), finish=AsyncMock(), close=AsyncMock(),
        )
        mcu = NS(
            zdt_stack_enabled=NS(get_value=AsyncMock(return_value=True)),
            zdt_start=NS(set=AsyncMock()), zdt_abort=NS(set=AsyncMock()),
            zdt_status_request=NS(set=AsyncMock()),
            zdt_status=NS(get_value=AsyncMock(
                return_value='{"run_id":7,"state":"complete","edges":2}'
            )),
        )
        default = ArhPresenter("arh")
        self.assertNotIn("due_z_stack", default.plan_map())
        uri = str(Path("tmp") / f"due-store-{uuid4().hex}.zarr")
        presenter = ArhPresenter(
            "arh", due_capture_factory=lambda: camera, due_store_uri=lambda _: uri
        )
        preview = []
        def receive_preview(readings):
            manifest = json.loads(manifest_path(uri, "due_external").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["frames"]), 2)
            preview.append(readings)

        presenter.preview_frame = receive_preview
        plan = presenter.plan_map()["due_z_stack"]["plan"]
        engine = RunEngine({}, context_managers=[])
        events = []
        starts = []
        engine.subscribe(lambda name, doc: events.append(doc["data"]) if name == "event" else None)
        engine.subscribe(lambda name, doc: starts.append(doc) if name == "start" else None)
        engine(plan(mcu, 7, 0, 1, 2, 0, 2, True, True))
        manifest = json.loads(manifest_path(uri, "due_external").read_text(encoding="utf-8"))
        timing = json.loads(timing_sidecar_path(uri, "due_external").read_text(encoding="utf-8"))
        self.assertEqual(starts[0]["capture_id"], manifest["run"]["run_id"])
        self.assertEqual(
            [entry["identity"]["producer_sequence"] for entry in manifest["frames"]], [1, 2]
        )
        self.assertEqual(
            [entry["service_monotonic_ns"] for entry in timing["records"]], [100, 200]
        )
        self.assertEqual(events, [{"trigger_edges": 2, "camera_frames": 2}])
        self.assertEqual(len(preview), 1)
        np.testing.assert_array_equal(preview[0]["dcam-buffer"]["value"], frames[-1].data)
        camera.close.assert_awaited_once()
        mcu.zdt_abort.set.assert_not_awaited()
