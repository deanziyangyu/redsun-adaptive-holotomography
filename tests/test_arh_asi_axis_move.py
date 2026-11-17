"""Regression coverage for single-axis point moves on the LX4000."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from redsun_aht.robotic.device.asi._stage import AsiXYStage
from redsun_aht.robotic.services.asi_stage import AsiAxisIO


class TestAsiAxisMove(unittest.IsolatedAsyncioTestCase):
    async def test_z_settling_checks_controller_counts(self):
        for actual, accepted in ((11.0, True), (11.1, False)):
            with self.subTest(actual=actual):
                client = AsyncMock()
                client.is_busy_z.return_value = False
                client.read_z_counts.side_effect = [0.0, actual, actual, 20.0]
                io = AsiAxisIO(
                    AsiXYStage(client), client, z_counts_per_um=10,
                    max_z_offset_um=1000, poll_period_s=0.05,
                )
                await io.capture_z_origin()
                attr = SimpleNamespace(io_ref=SimpleNamespace(axis="z"))
                if accepted:
                    await io.send(attr, 1.0)
                    self.assertEqual(io.outcome, "completed")
                    client.halt_z.assert_not_awaited()
                else:
                    with self.assertRaisesRegex(RuntimeError, "outside one count"):
                        await io.send(attr, 1.0)
                    self.assertEqual(io.outcome, "unknown")
                    client.halt_z.assert_awaited_once()
                    self.assertFalse(io.busy)
                    io._move_axis = AsyncMock(return_value=(1.0, 0.0))
                    await io.send(SimpleNamespace(io_ref=SimpleNamespace(axis="x")), 1.0)
                    await io.send(attr, 2.0)
                    self.assertEqual(io.outcome, "completed")
                self.assertEqual(client.move_z_counts.await_args_list[0].args, (10,))

    async def test_focus_stop_waits_for_idle_before_releasing_xy(self):
        client = AsyncMock()
        client.is_busy_z.side_effect = [False, False, False, True, False]
        client.read_z_counts.side_effect = [0.0, 14.0, 14.0]
        stage = AsiXYStage(client)
        stage.stop = AsyncMock()
        io = AsiAxisIO(stage, client, z_counts_per_um=10,
                       max_z_offset_um=1000, poll_period_s=0.05)
        await io.capture_z_origin()
        z = SimpleNamespace(io_ref=SimpleNamespace(axis="z"))
        x = SimpleNamespace(io_ref=SimpleNamespace(axis="x"))
        with self.assertRaisesRegex(RuntimeError, "outside one count"):
            await io.send(z, 1.0)
        self.assertFalse(io.busy)
        io._move_axis = AsyncMock(return_value=(1.0, 0.0))
        await io.send(x, 1.0)
        self.assertEqual(io.outcome, "completed")

    async def test_point_move_omits_partner_with_fractional_readback(self):
        for axis, before, settled in (
            ("x", (12.3, 3.5), (0, 3.5)),
            ("y", (3.5, 12.3), (3.5, 0)),
        ):
            with self.subTest(axis=axis):
                client = AsyncMock()
                client.read_xy_tenths.side_effect = [(0, 0), before, settled, settled, settled]
                client.is_busy_xy.return_value = False
                stage = AsiXYStage(client, max_xy_offset_um=1000, counts_per_um=10)
                await stage.connect()

                io = AsiAxisIO(
                    stage, client, z_counts_per_um=10,
                    max_z_offset_um=1000, poll_period_s=0.05,
                )
                await io.send(SimpleNamespace(io_ref=SimpleNamespace(axis=axis)), 0.0)

                client.move_axis_tenths.assert_awaited_once_with(axis.upper(), 0)
                client.move_xy_tenths.assert_not_called()
                self.assertEqual(io.outcome, "completed")

    async def test_point_move_rejects_target_beyond_startup_envelope(self):
        client = AsyncMock()
        client.read_xy_tenths.return_value = (100.0, -200.0)
        stage = AsiXYStage(client, max_xy_offset_um=1000, counts_per_um=10)
        await stage.connect()
        io = AsiAxisIO(
            stage, client, z_counts_per_um=10,
            max_z_offset_um=1000, poll_period_s=0.05,
        )
        for axis, target in (("x", 1010.1), ("x", -990.1), ("y", 980.1), ("y", -1020.1)):
            with self.subTest(axis=axis, target=target):
                attr = SimpleNamespace(io_ref=SimpleNamespace(axis=axis))
                with self.assertRaises(ValueError):
                    await io.send(attr, target)
        client.move_axis_tenths.assert_not_called()
        client.move_xy_tenths.assert_not_called()
