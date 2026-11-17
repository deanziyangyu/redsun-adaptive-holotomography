"""Rehearse the Due service's rotary commands without opening a serial port."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from redsun_aht.robotic.device.due_stack import DueStackStatus
from redsun_aht.robotic.services.unified_emmsh import EmmshController, EmmshIO


class RotarySerial:
    def __init__(self, reply: bytes = b"Roll: 8\n") -> None:
        self.requests: list[bytes] = []
        self.reply = reply

    async def exchange(self, request: bytes, *, terminator: bytes) -> bytes:
        if terminator != b"\n":
            raise AssertionError(terminator)
        self.requests.append(request)
        return self.reply


class TestRotaryService(unittest.IsolatedAsyncioTestCase):
    async def test_startup_respects_firmware_stop_state(self):
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
            b"ROTARY stopped\n", b"8\n", b"9\n", b"10\n",
            b"Exited Fixed Mode (Ready).\n",
        ]))
        limits = dict.fromkeys(("roll", "pitch", "yaw"), (2, 12))
        service = EmmshController(serial, "fixture", rotary_limits=limits)
        await service.connect()
        self.assertFalse(service.rotary_enabled.get())
        with self.assertRaisesRegex(RuntimeError, "disabled after a stop"):
            await service.move_rotary("roll", 8)
        self.assertEqual(serial.exchange.await_count, 5)

    async def test_stop_disables_all_rotary_targets_even_if_acknowledgement_is_lost(self):
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=TimeoutError))
        limits = dict.fromkeys(("roll", "pitch", "yaw"), (2, 12))
        service = EmmshController(serial, "fixture", rotary_limits=limits)
        await service.rotary_enabled.update(True)
        with self.assertRaises(TimeoutError):
            await service.safe_action("stop")
        self.assertFalse(service.rotary_enabled.get())
        await EmmshIO(service).update(service.rotary_enabled)
        self.assertFalse(service.rotary_enabled.get())
        for axis in limits:
            with self.assertRaisesRegex(RuntimeError, "disabled after a stop"):
                await service.move_rotary(axis, 8)
        serial.exchange.assert_awaited_once_with(b"k\n", terminator=b"\n")

    async def test_stopped_outputs_reject_each_target_without_retry(self) -> None:
        for axis, request in (("roll", b"r8\n"), ("pitch", b"t8\n"), ("yaw", b"y8\n")):
            with self.subTest(axis=axis):
                serial = RotarySerial(b"ERR rotary stopped\r\n")
                service = EmmshController(serial, "fixture", rotary_limits={axis: (2, 8)})
                with self.assertRaisesRegex(RuntimeError, "outputs are disabled after a stop"):
                    await service.move_rotary(axis, 8)
                self.assertEqual(serial.requests, [request])

    async def test_next_poll_reports_acknowledged_rotary_targets(self) -> None:
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
            b"Roll: 8\n", b"Pitch: 9\n", b"Yaw: 10\n",
            b"8\n", b"9\n", b"10\n",
        ]))
        limits = dict.fromkeys(("roll", "pitch", "yaw"), (2, 12))
        service = EmmshController(serial, "fixture", rotary_limits=limits)
        for axis, target in (("roll", 8), ("pitch", 9), ("yaw", 10)):
            await service.move_rotary(axis, target)
        state = json.loads(await service.snapshot())
        self.assertEqual(state["steps"], {"roll": 8, "pitch": 9, "yaw": 10})
        self.assertEqual(state["provenance"], "commanded_target")
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"r8\n", b"t9\n", b"y10\n", b"?r\n", b"?t\n", b"?y\n",
        ])

    async def test_reserved_stack_rejects_rotary_command_immediately(self) -> None:
        serial = RotarySerial(b"ZSTACK busy\r\n")
        service = EmmshController(serial, "fixture", rotary_limits={"roll": (2, 8)})
        with self.assertRaisesRegex(RuntimeError, "ZDT stack is reserved"):
            await service.move_rotary("roll", 8)
        self.assertEqual(serial.requests, [b"r8\n"])

    async def test_configured_targets_and_replies_for_each_axis(self) -> None:
        for axis, command in (("roll", b"r8\n"), ("pitch", b"t8\n"), ("yaw", b"y8\n")):
            with self.subTest(axis=axis):
                serial = RotarySerial(f"{axis.title()}: 8\n".encode("ascii"))
                service = EmmshController(serial, "fixture", rotary_limits={axis: (2, 8)})
                await service.move_rotary(axis, 8)
                self.assertEqual(serial.requests, [command])

    async def test_disabled_and_out_of_range_targets_do_not_write(self) -> None:
        serial = RotarySerial()
        disabled = EmmshController(serial, "fixture")
        limited = EmmshController(serial, "fixture", rotary_limits={"roll": (2, 8)})
        with self.assertRaises(ValueError):
            await disabled.move_rotary("roll", 8)
        for target in (1, 9, True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                await limited.move_rotary("roll", target)
        self.assertEqual(serial.requests, [])

    async def test_rotary_zero_publishes_all_axes_without_enabling_outputs(self):
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
            b"ROTARY zeroed\n", b"0\n", b"0\n", b"0\n",
        ]))
        service = EmmshController(serial, "fixture")
        service.rotary_stopped = True
        await EmmshIO(service).send(service.rotary_zero, True)
        self.assertEqual(json.loads(service.commanded_steps.get())["steps"], dict.fromkeys(("roll", "pitch", "yaw"), 0))
        self.assertTrue(service.rotary_stopped)
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"ROTARY ZERO\n", b"?r\n", b"?t\n", b"?y\n",
        ])

    async def test_rotary_zero_rejects_moving_axes(self):
        serial = RotarySerial(b"ROTARY moving\n")
        service = EmmshController(serial, "fixture")
        with self.assertRaisesRegex(RuntimeError, "ROTARY moving"):
            await service.zero_rotary()
        self.assertEqual(serial.requests, [b"ROTARY ZERO\n"])

    async def test_range_edit_publishes_immediately_without_latching_stop(self):
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
            b"ROTARY idle\n", b"0\n", b"0\n", b"0\n",
        ]))
        service = EmmshController(serial, "fixture", rotary_limits={"roll": (-114, 114)})
        await EmmshIO(service).send(service.rotary_range, "roll=-171:171")
        self.assertEqual(json.loads(service.commanded_steps.get())["rotary_limits"]["roll"], [-171, 171])
        self.assertFalse(service.rotary_stopped)
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"?i\n", b"?r\n", b"?t\n", b"?y\n",
        ])

    async def test_range_edit_refuses_while_rotary_axes_move(self):
        serial = RotarySerial(b"ROTARY moving\n")
        service = EmmshController(serial, "fixture", rotary_limits={"roll": (-114, 114)})
        with self.assertRaisesRegex(RuntimeError, "must be idle"):
            await service.set_rotary_range("roll=-171:171")
        self.assertEqual(service.rotary_limits["roll"], (-114, 114))

    async def test_bad_acknowledgement_fails_without_resending(self) -> None:
        serial = RotarySerial(b"Roll: 7\n")
        service = EmmshController(serial, "fixture", rotary_limits={"roll": (2, 8)})
        with self.assertRaisesRegex(RuntimeError, "Unexpected roll"):
            await service.move_rotary("roll", 8)
        self.assertEqual(serial.requests, [b"r8\n"])


class TestDueSoftwareDpct(unittest.IsolatedAsyncioTestCase):
    async def test_blank_accepts_idle_and_reserved_acknowledgments(self) -> None:
        for reply in (b"Exited Fixed Mode (Ready).\r\n", b"Panel blanked\r\n"):
            with self.subTest(reply=reply):
                serial = RotarySerial(reply)
                service = EmmshController(serial, "fixture", zdt_stack=True)
                await service.safe_action("blank")
                self.assertEqual(serial.requests, [b"q\n"])

        serial = RotarySerial(b"Stopped local axes\r\n")
        service = EmmshController(serial, "fixture", zdt_stack=True)
        with self.assertRaisesRegex(RuntimeError, "did not acknowledge blank"):
            await service.safe_action("blank")
        self.assertEqual(serial.requests, [b"q\n"])

    async def test_pattern_requires_opt_in_and_exact_acknowledgement(self) -> None:
        serial = RotarySerial(b"DPCT 13\n")
        service = EmmshController(serial, "fixture")
        with self.assertRaises(ValueError):
            await service.set_dpct_pattern(13)
        self.assertEqual(serial.requests, [])
        service = EmmshController(serial, "fixture", software_dpct=True)
        service.post_initialise()
        await service.set_dpct_pattern(13)
        self.assertEqual(serial.requests, [b"DPCT 13\n"])
        serial.reply = b"DPCT 14\n"
        with self.assertRaisesRegex(RuntimeError, "acknowledge"):
            await service.set_dpct_pattern(13)
        self.assertEqual(serial.requests, [b"DPCT 13\n"] * 2)


class TestDueXStackStartup(unittest.IsolatedAsyncioTestCase):
    async def test_software_stack_selects_mode_before_arming_and_acknowledges_plane(self):
        serial = SimpleNamespace(
            exchange=AsyncMock(side_effect=[
                b"ZSTACK committed\n", b"ZSTACK software\n", b"ZA pending\n",
                b"ZA pending\n", b"ZSTACK armed_host_asserted\n", b"ZSTACK started\n",
                b"ZSTACK waiting_capture index=0\n", b"ZSTACK captured\n",
            ]),
            read_line=AsyncMock(side_effect=[b"ZA config_verified=1\n", b"ZA state_verified=1\n"]),
        )
        service = EmmshController(
            serial, "fixture", zdt_stack=True, zdt_software=True, zdt_software_only=True,
        )
        request = json.dumps({
            "run_id": 7, "first": 0, "step": 1, "count": 2,
            "lower": 0, "upper": 2, "camera_ready": True, "software": True,
        })
        with self.assertRaisesRegex(ValueError, "software capture mode is unavailable"):
            await service.start_zdt_stack(request)
        serial.exchange.assert_not_awaited()
        await service.zdt_software_enabled.update(True)
        external_request = json.loads(request)
        del external_request["software"]
        with self.assertRaisesRegex(ValueError, "requires software capture"):
            await service.start_zdt_stack(json.dumps(external_request))
        serial.exchange.assert_not_awaited()
        await service.start_zdt_stack(request)
        await service.query_zdt_stack(7)
        self.assertEqual(DueStackStatus.parse(service.zdt_status.get(), 7, 2),
                         DueStackStatus(7, "waiting_capture", 0))
        await service.acknowledge_zdt_capture("7 0")
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"COMMIT 7 0 1 2 0 2 1\n", b"SOFTWARE 7\n", b"zacfg?\n", b"zastate?\n",
            b"ARM 7\n", b"START 7\n", b"STATUS 7\n", b"CAPTURED 7 0\n",
        ])

    async def test_shutdown_releases_idle_coils_and_stops_reserved_work(self):
        for probe, command, reply in (
            (b"ROTARY idle\n", b"w\n", b"Rotary outputs off\n"),
            (b"ZSTACK busy\n", b"k\n", b"Stopped local axes\n"),
        ):
            with self.subTest(probe=probe):
                serial = SimpleNamespace(
                    exchange=AsyncMock(side_effect=[probe, reply, b"Panel blanked\n"]), close=Mock(),
                )
                service = EmmshController(serial, "fixture", zdt_stack=True)
                await service.close()
                self.assertEqual([call.args[0] for call in serial.exchange.await_args_list],
                                 [b"?i\n", command, b"q\n"])
                serial.close.assert_called_once()

    async def test_positioner_jog_releases_coils_after_controller_idle(self):
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
            b"ROTARY idle\n", b"Enabled\n", b"Pitch: -4\n",
            b"ROTARY moving\n", b"ROTARY idle\n", b"Rotary outputs off\n",
        ]))
        service = EmmshController(serial, "fixture", rotary_limits={"pitch": (-114, 114)})
        await service.jog_rotary("pitch=-4")
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"?i\n", b"j\n", b"t-4\n", b"?i\n", b"?i\n", b"w\n",
        ])
        self.assertFalse(service.rotary_enabled.get())

    async def test_signed_rotary_target_is_sent_and_read_back(self):
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
            b"Roll: -114\n", b"-114\n", b"0\n", b"0\n",
        ]))
        service = EmmshController(serial, "fixture", rotary_limits={"roll": (-114, 114)})
        await service.move_rotary("roll", -114)
        self.assertEqual(serial.exchange.await_args_list[0].args[0], b"r-114\n")
        self.assertEqual(json.loads(await service.snapshot())["steps"]["roll"], -114)
        with self.assertRaises(ValueError):
            await service.move_rotary("roll", -115)
        self.assertEqual(serial.exchange.await_count, 4)

    async def test_zdt_jog_uses_fresh_origin_and_software_mode(self):
        serial = SimpleNamespace(
            exchange=AsyncMock(side_effect=[
                b"ZA pending\n", b"ZSTACK committed\n",
                b"ZA pending\n", b"ZA pending\n", b"ZSTACK armed_host_asserted\n",
                b"ZSTACK started\n",
            ]),
            read_line=AsyncMock(side_effect=[
                b"ZA state_verified=1 position_tenths_deg=20\n", b"ZA config_verified=1\n",
                b"ZA state_verified=1 position_tenths_deg=20\n",
            ]),
        )
        service = EmmshController(serial, "fixture", zdt_stack=True, zdt_software=True)
        await service.zdt_software_enabled.update(True)
        await service.start_zdt_stack(json.dumps(dict(run_id=7, jog=-5, camera_ready=True, software=True)))
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"zastate?\n", b"JOG 7 15 0 1 -3580 3600 1\n", b"zacfg?\n",
            b"zastate?\n", b"ARM 7\n", b"START 7\n",
        ])

    async def test_absolute_zdt_jog_preserves_target_when_origin_changes(self):
        serial = SimpleNamespace(
            exchange=AsyncMock(side_effect=[
                b"ZA pending\n", b"ZSTACK committed\n", b"ZA pending\n",
                b"ZA pending\n", b"ZSTACK armed_host_asserted\n", b"ZSTACK started\n",
            ]),
            read_line=AsyncMock(side_effect=[
                b"ZA state_verified=1 position_tenths_deg=21\n", b"ZA config_verified=1\n",
                b"ZA state_verified=1 position_tenths_deg=20\n",
            ]),
        )
        service = EmmshController(serial, "fixture", zdt_stack=True, zdt_software=True)
        await service.zdt_software_enabled.update(True)
        await service.start_zdt_stack(json.dumps(dict(run_id=7, jog_target=-500, camera_ready=True, software=True)))
        self.assertEqual(serial.exchange.await_args_list[1].args[0], b"JOG 7 -500 0 1 -3579 3600 1\n")

    async def test_zdt_read_retries_transient_state_rejection(self):
        serial = SimpleNamespace(
            exchange=AsyncMock(return_value=b"ZA pending\n"),
            read_line=AsyncMock(side_effect=[
                b"ZA state_verified=0\n", b"ZA state_frame=010203\n",
                b"ZA state_verified=1 position_tenths_deg=-84\n",
            ]),
        )
        service = EmmshController(serial, "fixture", zdt_stack=True)
        await service.query_zdt_stack(0)
        self.assertEqual(serial.exchange.await_count, 2)
        self.assertAlmostEqual(json.loads(service.zdt_status.get())["position_um"], -84 / 3.6)

    async def test_completed_zdt_jog_does_not_repeat_admission_check(self):
        serial = SimpleNamespace(exchange=AsyncMock(return_value=b"ZSTACK state=2 edges=1\n"))
        service = EmmshController(serial, "fixture", zdt_stack=True)
        service._zdt_jog = (7, 15)
        await service.query_zdt_stack(7)
        status = json.loads(service.zdt_status.get())
        self.assertAlmostEqual(status["target_um"], 15 / 3.6)
        self.assertNotIn("position_um", status)
        serial.exchange.assert_awaited_once_with(b"STATUS 7\n", terminator=b"\n")

    async def test_missing_commit_reply_still_aborts_uncertain_run(self):
        serial = SimpleNamespace(exchange=AsyncMock(side_effect=[
            TimeoutError("COMMIT reply lost"), b"ZSTACK cancelled\n",
        ]))
        service = EmmshController(serial, "fixture", zdt_stack=True)
        request = json.dumps({
            "run_id": 7, "first": 0, "step": 1, "count": 2,
            "lower": 0, "upper": 2, "camera_ready": True,
        })
        with self.assertRaisesRegex(TimeoutError, "COMMIT reply lost"):
            await service.start_zdt_stack(request)
        self.assertEqual(
            [call.args[0] for call in serial.exchange.await_args_list],
            [b"COMMIT 7 0 1 2 0 2 1\n", b"ABORT 7\n"],
        )

    async def test_explicit_commit_rejection_preserves_reason_and_sends_no_abort(self):
        serial = RotarySerial(b"ZSTACK error=rotary_stopped\r\n")
        service = EmmshController(serial, "fixture", zdt_stack=True)
        request = json.dumps({
            "run_id": 7, "first": 0, "step": 1, "count": 2,
            "lower": 0, "upper": 2, "camera_ready": True,
        })
        with self.assertRaisesRegex(RuntimeError, "ZSTACK error=rotary_stopped"):
            await service.start_zdt_stack(request)
        self.assertEqual(serial.requests, [b"COMMIT 7 0 1 2 0 2 1\n"])

    async def test_verified_state_allows_arm_then_start(self) -> None:
        serial = SimpleNamespace(
            exchange=AsyncMock(side_effect=[
                b"ZSTACK committed\n", b"ZSTACK EDGE 2\n", b"ZA pending\n",
                b"ZSTACK armed_host_asserted\n", b"ZSTACK started\n",
            ]),
            read_line=AsyncMock(side_effect=[
                b"ZA pending\n", b"ZSTACK complete edges=2\n", b"ZA config_verified=1\n",
                b"ZA state_verified=1\n",
            ]),
        )
        service = EmmshController(serial, "fixture", zdt_stack=True)
        request = json.dumps({
            "run_id": 7, "first": 0, "step": 1, "count": 2,
            "lower": 0, "upper": 2, "camera_ready": True,
        })
        await service.start_zdt_stack(request)
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"COMMIT 7 0 1 2 0 2 1\n", b"zacfg?\n", b"zastate?\n", b"ARM 7\n", b"START 7\n",
        ])

    async def test_failed_state_verification_aborts_without_arming(self) -> None:
        serial = SimpleNamespace(
            exchange=AsyncMock(side_effect=[
                b"ZSTACK committed\n", b"ZA pending\n",
                b"ZA pending\n", b"ZSTACK cancelled\n",
            ]),
            read_line=AsyncMock(side_effect=[b"ZA config_verified=1\n", b"ZA state_verified=0\n"]),
        )
        service = EmmshController(serial, "fixture", zdt_stack=True)
        request = json.dumps({
            "run_id": 7, "first": 0, "step": 1, "count": 2,
            "lower": 0, "upper": 2, "camera_ready": True,
        })
        with self.assertRaisesRegex(RuntimeError, "did not verify ZDT state"):
            await service.start_zdt_stack(request)
        self.assertEqual([call.args[0] for call in serial.exchange.await_args_list], [
            b"COMMIT 7 0 1 2 0 2 1\n", b"zacfg?\n", b"zastate?\n", b"ABORT 7\n",
        ])

    async def test_cancelled_stack_status_preserves_run_id(self) -> None:
        serial = RotarySerial(b"ZSTACK state=6 edges=0\r\n")
        service = EmmshController(serial, "fixture", zdt_stack=True)
        await service.query_zdt_stack(7)
        status = DueStackStatus.parse(service.zdt_status.get(), 7, 2)
        self.assertEqual(status, DueStackStatus(7, "cancelled", 0))
        self.assertEqual(serial.requests, [b"STATUS 7\n"])

    async def test_emm_identity_rejects_x_stack_before_other_commands(self) -> None:
        serial = RotarySerial(b"ARH-DUE EMM protocol=1\r\n")
        service = EmmshController(serial, "fixture", zdt_stack=True)
        with self.assertRaisesRegex(RuntimeError, "Expected Due ZDT-X protocol 1"):
            await service.connect()
        self.assertEqual(serial.requests, [b"ID?\n"])

    async def test_x_identity_requires_separate_stack_capability(self) -> None:
        serial = RotarySerial(b"ARH-DUE ZDT-X protocol=1\r\n")
        service = EmmshController(serial, "fixture", zdt_stack=True)
        with self.assertRaisesRegex(RuntimeError, "lacks the commissioned X-stack interface"):
            await service.connect()
        self.assertEqual(serial.requests, [b"ID?\n", b"ZSTACK?\n"])
