"""Fake serial tests for the experimental single-owner unified MCU transport."""

import asyncio

import pytest

from redsun_aht.device.unified_mcu import (
    AngularStack,
    UnifiedMcu,
    UnifiedMcuError,
    open_unified_port,
    parse_x_state,
)


class FakePort:
    def __init__(self, *, complete=True, wrong_config=False, repeat=False):
        self.is_open = True
        self.commands = []
        self.responses = asyncio.Queue()
        self.started = asyncio.Event()
        self.complete = complete
        self.wrong_config = wrong_config
        self.repeat = repeat
        self.auto = True
        self.x_position = 0

    def emit(self, line):
        self.responses.put_nowait((line + "\n").encode())

    async def write(self, data):
        cmd = data.decode().strip()
        self.commands.append(cmd)
        if cmd == "identity":
            self.emit("ARH-UNIFIED-STACK/0.18")
        elif cmd in {"SHOTHOLD 0", "SHOTHOLD 1"}:
            self.emit("SHOT hold=" + cmd[-1])
        elif cmd in {"SHOTBIND 0", "SHOTBIND 1"}:
            self.emit("SHOT binding=" + cmd[-1])
        elif cmd.startswith("COMMIT "):
            self.emit("ZSTACK committed")
        elif cmd.startswith("TRIGMASK "):
            self.emit("ZSTACK trigger_mask=" + cmd.split()[2])
        elif cmd == "zasyncxconfig":
            b = bytearray(37)
            b[0:4] = bytes([1, 0x42, 37, 24])
            b[25] = 1
            b[26] = int(self.wrong_config)
            b[36] = 0x6B
            self.emit("ZA pending")
            self.emit("ZA config_hex=" + b.hex())
        elif cmd == "zasyncxstate":
            b = bytearray(37)
            b[0:4] = bytes([1, 0x43, 37, 12])
            b[14] = b[22] = int(self.x_position < 0)
            b[15:19] = b[23:27] = abs(self.x_position).to_bytes(4, "big")
            b[34:36] = bytes([3, 3])
            b[36] = 0x6B
            self.emit("ZA pending")
            self.emit("ZA state_hex=" + b.hex())
        elif cmd.startswith("ARM "):
            self.emit("ZSTACK armed_host_asserted" if self.auto else "ZSTACK error=arm")
        elif cmd.startswith("START "):
            self.emit("ZSTACK started")
            self.started.set()
            if self.complete:
                self.emit("ZSTACK EDGE 0 0")
                self.emit("ZSTACK EDGE 1 18")
                self.emit("ZSTACK EDGE 2 36")
                self.emit("ZSTACK complete edges=3")
        elif cmd.startswith("ABORT "):
            self.emit("ZSTACK stopping")
            self.emit("ZSTACK stop_unverified edges=0 reset_required")
        elif cmd in {
            "00",
            "01",
            "11",
            "12",
            "13",
            "14",
            "15",
            "51",
            "52",
            "53",
            "61",
            "62",
            "63",
            "64",
        }:
            self.emit(str(int(cmd)))
        elif cmd in {"c0", "c1"}:
            self.auto = cmd == "c1"
            self.emit(
                "Camera trigger enabled." if self.auto else "Camera trigger disabled."
            )
        elif cmd in {"d1", "d2", "d3"}:
            self.emit(
                {"d1": "Triggered Cam", "d2": "Triggered Dev2", "d3": "Triggered Dev3"}[
                    cmd
                ]
            )
        else:
            raise AssertionError(cmd)
        return len(data)

    async def flush(self):
        pass

    async def read_until(self, expected=b"\n"):
        return await self.responses.get()

    def close(self):
        self.is_open = False


def definition():
    return AngularStack(7, 0, 18, 3, -180, 180)


async def test_color_and_brightness_commands_are_acknowledged():
    port = FakePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    for command in ("51", "52", "53", "61", "62", "63", "64"):
        await owner.illumination(command)
    assert port.commands[1:] == ["51", "52", "53", "61", "62", "63", "64"]
    with pytest.raises(ValueError):
        await owner.illumination("65")


def test_x_state_decode_preserves_signed_units_and_rejects_bad_sign():
    frame = bytearray(37)
    frame[:4] = bytes([1, 0x43, 37, 12])
    frame[14], frame[15:19] = 1, (36).to_bytes(4, "big")
    frame[19], frame[20:22] = 1, (12).to_bytes(2, "big")
    frame[22], frame[23:27] = 0, (18).to_bytes(4, "big")
    frame[27], frame[28:32] = 1, (5).to_bytes(4, "big")
    frame[34:37] = bytes([3, 3, 0x6B])
    state = parse_x_state(bytes(frame))
    assert (state.target_tenths, state.position_tenths) == (-36, 18)
    assert (state.speed_tenths_rpm, state.error_hundredths_degree) == (-12, -5)
    assert (state.home_flags, state.motor_flags, state.quality) == (
        3,
        3,
        "controller_reported",
    )
    frame[14] = 2
    with pytest.raises(UnifiedMcuError, match="invalid"):
        parse_x_state(bytes(frame))


async def test_read_x_state_is_serialized_and_failed_query_quarantines():
    class BadStatePort(FakePort):
        async def write(self, data):
            if data == b"zasyncxstate\n":
                self.commands.append("zasyncxstate")
                self.emit("ZA pending")
                self.emit("ZA state_hex=bad")
                return len(data)
            return await super().write(data)

    port = FakePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    state = await owner.read_x_state()
    assert state.position_tenths == 0 and state.motor_flags == 3
    assert port.commands == ["identity", "zasyncxstate"]
    bad_port = BadStatePort()
    bad = UnifiedMcu(bad_port)
    await bad.identify()
    with pytest.raises(UnifiedMcuError, match="hex"):
        await bad.read_x_state()
    assert bad.uncertain
    with pytest.raises(UnifiedMcuError, match="uncertain"):
        await bad.read_x_state()
    assert bad_port.commands.count("zasyncxstate") == 1


def test_open_sets_modem_lines_before_open(monkeypatch):
    class Port:
        dtr = True
        rts = True
        opened = False

        def open(self):
            assert self.dtr is False and self.rts is False
            self.opened = True

    port = Port()

    def fake_open(name, **kwargs):
        assert name == "COM-test" and kwargs["do_not_open"] is True
        assert kwargs["baudrate"] == 115200
        return port

    monkeypatch.setattr("redsun_aht.device.unified_mcu.serial_for_url", fake_open)
    assert open_unified_port("COM-test") is port
    assert port.opened


def test_stack_rejects_integer_overflow_even_for_one_plane():
    with pytest.raises(ValueError, match="integer range"):
        AngularStack(7, 0, 2_147_483_648, 1, -180, 180)


async def test_unified_port_identifies_and_runs_one_stack():
    port = FakePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    await owner.illumination("00")
    result = await owner.stack(definition())
    assert result.targets_tenths == (0, 18, 36)
    assert owner.progress is not None
    assert owner.progress.targets_tenths == (0, 18, 36)
    assert owner.progress.complete
    assert port.commands.count("START 7") == 1
    assert port.commands == [
        "identity",
        "00",
        definition().commit_line(),
        "zasyncxconfig",
        "zasyncxstate",
        "ARM 7",
        "START 7",
    ]
    await owner.stack(definition())
    assert port.commands.count("START 7") == 2


async def test_mask_firmware_keeps_default_output_and_repeat_behavior():
    class MaskPort(FakePort):
        async def write(self, data):
            if data == b"identity\n":
                self.commands.append("identity")
                self.emit("ARH-UNIFIED-STACK/0.18")
                return len(data)
            return await super().write(data)

    port = MaskPort(repeat=True)
    owner = UnifiedMcu(port)
    await owner.identify()
    await owner.stack(definition())
    await owner.stack(AngularStack(8, 0, 18, 3, -180, 180))
    assert port.commands.count("START 7") == 1
    assert port.commands.count("START 8") == 1
    assert not any(command.startswith("TRIGMASK") for command in port.commands)


async def test_nondefault_trigger_mask_requires_supported_firmware():
    stack = AngularStack(7, 0, 18, 3, -180, 180, trigger_mask=6)

    class MaskPort(FakePort):
        async def write(self, data):
            if data == b"identity\n":
                self.commands.append("identity")
                self.emit("ARH-UNIFIED-STACK/0.18")
                return len(data)
            return await super().write(data)

    port = MaskPort(repeat=True)
    owner = UnifiedMcu(port)
    await owner.identify()
    await owner.stack(stack)
    assert (
        port.commands.index("TRIGMASK 7 6")
        == port.commands.index(stack.commit_line()) + 1
    )
    assert port.commands.index("TRIGMASK 7 6") < port.commands.index("ARM 7")


def test_stack_rejects_invalid_trigger_mask():
    for mask in (0, 8, True):
        with pytest.raises(ValueError):
            AngularStack(7, 0, 18, 3, -180, 180, trigger_mask=mask)


async def test_fragmented_serial_responses_are_assembled():
    class FragmentedPort(FakePort):
        remainder = b""

        async def read_until(self, expected=b"\n"):
            if not self.remainder:
                self.remainder = await super().read_until(expected)
            part, self.remainder = self.remainder[:3], self.remainder[3:]
            return part

    port = FragmentedPort()
    owner = UnifiedMcu(port)
    await owner.identify()
    assert (await owner.stack(definition())).targets_tenths == (0, 18, 36)
    assert port.commands.count("START 7") == 1


@pytest.mark.parametrize("chunks", [[b"x" * 200, b"y" * 57], [b"bad\nextra\n"]])
async def test_invalid_fragmented_line_quarantines_owner(chunks):
    port = FakePort()
    owner = UnifiedMcu(port, line_timeout_s=0.02)
    for chunk in chunks:
        port.responses.put_nowait(chunk)
    with pytest.raises(UnifiedMcuError, match="invalid serial line"):
        await owner.identify()
    with pytest.raises(UnifiedMcuError, match="uncertain"):
        await owner.identify()


async def test_incomplete_line_times_out_and_quarantines_owner():
    class SilentPort(FakePort):
        async def write(self, data):
            return len(data)

    port = SilentPort()
    port.responses.put_nowait(b"partial")
    owner = UnifiedMcu(port, line_timeout_s=0.02)
    with pytest.raises(TimeoutError):
        await owner.identify()
    with pytest.raises(UnifiedMcuError, match="uncertain"):
        await owner.identify()


async def test_non_ascii_line_quarantines_owner():
    port = FakePort()
    port.responses.put_nowait(b"invalid\xff\n")
    owner = UnifiedMcu(port)
    with pytest.raises(UnicodeDecodeError):
        await owner.identify()
    with pytest.raises(UnifiedMcuError, match="uncertain"):
        await owner.identify()


async def test_abort_uses_same_owner_and_blocks_illumination():
    port = FakePort(complete=False)
    owner = UnifiedMcu(port)
    await owner.identify()
    task = asyncio.create_task(owner.stack(definition()))
    await asyncio.wait_for(port.started.wait(), 1)
    with pytest.raises(UnifiedMcuError, match="busy"):
        await owner.illumination("11")
    with pytest.raises(UnifiedMcuError, match="busy"):
        await owner.read_x_state()
    assert port.commands.count("zasyncxstate") == 1  # stack preflight only
    await owner.abort(7)
    with pytest.raises(UnifiedMcuError, match="unverified"):
        await task
    assert port.commands.count("ABORT 7") == 1
    with pytest.raises(UnifiedMcuError, match="uncertain"):
        await owner.illumination("00")


async def test_unsupported_scaling_refuses_before_start():
    port = FakePort(wrong_config=True)
    owner = UnifiedMcu(port)
    await owner.identify()
    with pytest.raises(UnifiedMcuError, match="configuration"):
        await owner.stack(definition())
    assert not any(command.startswith("START") for command in port.commands)


async def test_completed_stack_can_reuse_id():
    port = FakePort(repeat=True)
    owner = UnifiedMcu(port)
    await owner.identify()
    await owner.stack(definition())
    await owner.stack(definition())
    assert port.commands.count("START 7") == 2


async def test_partial_events_survive_abort_and_are_not_marked_complete():
    port = FakePort(complete=False)
    owner = UnifiedMcu(port)
    await owner.identify()
    task = asyncio.create_task(owner.stack(definition()))
    await asyncio.wait_for(port.started.wait(), 1)
    port.emit("ZSTACK EDGE 0 0")
    port.emit("ZSTACK EDGE 1 18")
    await owner.abort(7)
    with pytest.raises(UnifiedMcuError, match="unverified"):
        await task
    assert owner.progress is not None
    assert owner.progress.run_id == 7
    assert owner.progress.targets_tenths == (0, 18)
    assert not owner.progress.complete
    assert port.commands.count("START 7") == 1
    assert port.commands.count("ABORT 7") == 1


async def test_cancellation_never_resends_start_and_requests_one_abort():
    port = FakePort(complete=False)
    owner = UnifiedMcu(port, line_timeout_s=0.05)
    await owner.identify()
    # Abort promptly rather than waiting for the production per-plane deadline.
    task = asyncio.create_task(owner.stack(definition()))
    await asyncio.wait_for(port.started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert port.commands.count("START 7") == 1
    assert port.commands.count("ABORT 7") == 1


class ShotFakePort(FakePort):
    def __init__(self):
        super().__init__(repeat=True)
        self.shot_step = 0
        self.per_plane = 0
        self.first = 0
        self.bad_ready = False

    async def write(self, data):
        cmd = data.decode().strip()
        if cmd == "identity":
            self.commands.append(cmd)
            self.emit("ARH-UNIFIED-STACK/0.18")
            return len(data)
        if cmd.startswith("SHOTCFG "):
            self.commands.append(cmd)
            args = list(map(int, cmd.split()[1:]))
            self.first, self.shot_step, self.per_plane = args[1], args[2], args[4]
            self.emit("SHOT configured")
            return len(data)
        if cmd.startswith("SHOTARM "):
            self.commands.append(cmd)
            self.emit("SHOT armed_host_asserted")
            return len(data)
        if cmd.startswith("SHOT "):
            self.commands.append(cmd)
            run, index, pattern = map(int, cmd.split()[1:])
            target = self.first + self.shot_step * (index // self.per_plane)
            self.emit(
                f"SHOT READY {run} {index} {target + int(self.bad_ready)} {pattern}"
            )
            return len(data)
        if cmd.startswith("SHOTFIN "):
            self.commands.append(cmd)
            self.emit("SHOT complete")
            return len(data)
        if cmd.startswith("SHOTABORT "):
            self.commands.append(cmd)
            return len(data)
        return await super().write(data)


async def test_software_shots_wait_for_each_host_frame_before_advancing():
    port = ShotFakePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    observed = []

    async def snap(index, plane, pattern, target):
        assert port.commands[-1] == f"SHOT 9 {index} {pattern}"
        observed.append((index, plane, pattern, target))

    await owner.software_shots(
        AngularStack(9, 0, 18, 2, -180, 180),
        4,
        [1] * 8,
        snap,
        camera_exclusive=True,
        illumination_ready=True,
    )
    assert observed == [(i, i // 4, 1, 0 if i < 4 else 18) for i in range(8)]
    assert port.commands.count("SHOTCFG 9 0 18 2 4 -180 180") == 1
    assert port.commands[-1] == "SHOTFIN 9 7"
    assert not owner.uncertain


async def test_software_shots_snap_failure_aborts_without_next_shot():
    port = ShotFakePort()
    owner = UnifiedMcu(port)
    await owner.identify()

    async def snap(index, plane, pattern, target):
        raise TimeoutError("camera frame uncertain")

    with pytest.raises(TimeoutError, match="camera frame"):
        await owner.software_shots(
            AngularStack(9, 0, 18, 2, -180, 180),
            4,
            [1] * 8,
            snap,
            camera_exclusive=True,
            illumination_ready=True,
        )
    assert port.commands[-2:] == ["SHOT 9 0 1", "SHOTABORT 9"]
    assert owner.uncertain


async def test_software_shots_bad_ready_never_snaps_and_aborts():
    port = ShotFakePort()
    port.bad_ready = True
    owner = UnifiedMcu(port)
    await owner.identify()
    called = []

    async def snap(*args):
        called.append(args)

    with pytest.raises(UnifiedMcuError, match="readiness"):
        await owner.software_shots(
            AngularStack(9, 0, 18, 2, -180, 180),
            4,
            [1] * 8,
            snap,
            camera_exclusive=True,
            illumination_ready=True,
        )
    assert not called and port.commands[-1] == "SHOTABORT 9" and owner.uncertain


async def test_software_shots_failed_preflight_requests_blank_without_arm():
    port = ShotFakePort()
    port.wrong_config = True
    owner = UnifiedMcu(port)
    await owner.identify()

    async def snap(*args):
        raise AssertionError("must not snap")

    with pytest.raises(UnifiedMcuError, match="configuration"):
        await owner.software_shots(
            AngularStack(9, 0, 18, 2, -180, 180),
            4,
            [1] * 8,
            snap,
            camera_exclusive=True,
            illumination_ready=True,
        )
    assert port.commands[-1] == "00"
    assert not any(c.startswith("SHOTARM") for c in port.commands)
    assert owner.uncertain


class AutoShotFakePort(ShotFakePort):
    def __init__(self, bad_index=None):
        super().__init__()
        self.bad_index = bad_index
        self.planes = 0

    async def write(self, data):
        cmd = data.decode().strip()
        if cmd == "identity":
            self.commands.append(cmd)
            self.emit("ARH-UNIFIED-STACK/0.18")
            return len(data)
        if cmd.startswith("SHOTCFG "):
            self.planes = int(cmd.split()[4])
        if cmd.startswith("SHOTAUTO "):
            self.commands.append(cmd)
            total = self.planes * self.per_plane
            self.emit("SHOT auto_started_host_asserted")
            self.emit(
                f"SHOT complete attempts={total - int(self.bad_index is not None)}"
            )
            return len(data)
        return await super().write(data)


async def test_automatic_shots_reports_attempts_not_frames():
    port = AutoShotFakePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    summary = await owner.automatic_shots(
        AngularStack(9, 0, 18, 2, -180, 180, 2),
        4,
        3,
        50,
        camera_armed=True,
        illumination_ready=True,
    )
    assert summary.attempted_pulses == 8
    assert port.commands[-1] == "SHOTAUTO 9 3 2 50"
    assert not owner.uncertain


async def test_automatic_shots_bad_completion_aborts():
    port = AutoShotFakePort(bad_index=3)
    owner = UnifiedMcu(port)
    await owner.identify()
    with pytest.raises(UnifiedMcuError, match="unexpected shot completion"):
        await owner.automatic_shots(
            AngularStack(9, 0, 18, 2, -180, 180),
            4,
            1,
            50,
            camera_armed=True,
            illumination_ready=True,
        )
    assert port.commands[-1] == "SHOTABORT 9" and owner.uncertain


async def test_runtime_motion_profile_read_set_and_validation():
    class ProfilePort(FakePort):
        async def write(self, data):
            cmd = data.decode().strip()
            if cmd == "identity":
                self.commands.append(cmd)
                self.emit("ARH-UNIFIED-STACK/0.18")
            elif cmd in ("zprofile?", "zprofile 250 800"):
                self.commands.append(cmd)
                self.emit("ZPROFILE speed_tenths_rpm=250 accel_rpm_s=800")
            else:
                return await super().write(data)
            return len(data)

    port = ProfilePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    assert await owner.motion_profile() == (250, 800)
    assert await owner.motion_profile(250, 800) == (250, 800)
    with pytest.raises(ValueError):
        await owner.motion_profile(0, 800)
    assert port.commands == ["identity", "zprofile?", "zprofile 250 800"]


class TransitionalStatePort(FakePort):
    def __init__(self, states):
        super().__init__()
        self.states = list(states)
        self.state_reads = 0

    async def write(self, data):
        if data == b"zasyncxstate\n":
            self.commands.append("zasyncxstate")
            speed, motor_flags = self.states[
                min(self.state_reads, len(self.states) - 1)
            ]
            self.state_reads += 1
            frame = bytearray(37)
            frame[:4] = bytes([1, 0x43, 37, 12])
            frame[20:22] = speed.to_bytes(2, "big")
            frame[34:36] = bytes([3, motor_flags])
            frame[36] = 0x6B
            self.emit("ZA pending")
            self.emit("ZA state_hex=" + frame.hex())
            return len(data)
        return await super().write(data)


async def test_between_run_preflight_waits_for_stationary_then_starts_once():
    port = TransitionalStatePort([(2, 3), (1, 3), (0, 3)])
    owner = UnifiedMcu(port)
    await owner.identify()
    result = await owner.stack(definition())
    assert result.targets_tenths == (0, 18, 36)
    assert port.state_reads == 3
    assert port.commands.count("COMMIT 7 0 18 3 -180 180 1") == 1
    assert port.commands.count("START 7") == 1


async def test_between_run_preflight_fault_refuses_without_start():
    port = TransitionalStatePort([(0, 7)])
    owner = UnifiedMcu(port)
    await owner.identify()
    with pytest.raises(UnifiedMcuError, match="controller fault"):
        await owner.stack(definition())
    assert port.state_reads == 1 and not any(
        c.startswith("START ") for c in port.commands
    )
    assert owner.uncertain


async def test_between_run_preflight_timeout_never_starts():
    port = TransitionalStatePort([(1, 3)])
    owner = UnifiedMcu(port)
    await owner.identify()
    with pytest.raises(UnifiedMcuError, match="not stationary"):
        await owner.stack(definition())
    assert port.state_reads > 1 and not any(
        c.startswith("START ") for c in port.commands
    )
    assert owner.uncertain


async def test_runtime_readiness_error_limit():
    class TolerancePort(FakePort):
        async def write(self, data):
            cmd = data.decode().strip()
            if cmd == "identity":
                self.commands.append(cmd)
                self.emit("ARH-UNIFIED-STACK/0.18")
            elif cmd in ("zreadyerr?", "zreadyerr 20", "zreadyerr 0"):
                self.commands.append(cmd)
                self.emit(
                    "ZREADYERR hundredths=" + ("20" if cmd != "zreadyerr 0" else "0")
                )
            else:
                return await super().write(data)
            return len(data)

    port = TolerancePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    assert await owner.readiness_error_limit() == 20
    assert await owner.readiness_error_limit(20) == 20
    assert await owner.readiness_error_limit(0) == 0
    with pytest.raises(ValueError):
        await owner.readiness_error_limit(-1)


class OptionalStagePort(ShotFakePort):
    async def write(self, data):
        cmd = data.decode().strip()
        if cmd == "identity":
            self.commands.append(cmd)
            self.emit("ARH-UNIFIED-STACK/0.18")
            return len(data)
        if cmd in ("SHOTBIND 0", "SHOTBIND 1"):
            self.commands.append(cmd)
            self.emit("SHOT binding=" + cmd[-1])
            return len(data)
        return await super().write(data)


class HoldStagePort(OptionalStagePort):
    async def write(self, data):
        cmd = data.decode().strip()
        if cmd == "identity":
            self.commands.append(cmd)
            self.emit("ARH-UNIFIED-STACK/0.18")
            return len(data)
        if cmd in ("SHOTHOLD 0", "SHOTHOLD 1"):
            self.commands.append(cmd)
            self.emit("SHOT hold=" + cmd[-1])
            return len(data)
        return await super().write(data)


async def test_software_shot_hold_negotiation_and_reset():
    port = HoldStagePort()
    owner = UnifiedMcu(port)
    await owner.identify()

    async def snap(*args):
        pass

    for run_id, hold in ((9, True), (10, False)):
        await owner.software_shots(
            AngularStack(run_id, 0, 18, 1, -180, 180),
            1,
            [1],
            snap,
            camera_exclusive=True,
            illumination_ready=True,
            bind_stage=False,
            hold_illumination=hold,
        )
    assert [cmd for cmd in port.commands if cmd.startswith("SHOTHOLD")] == [
        "SHOTHOLD 1",
        "SHOTHOLD 0",
    ]


async def test_lost_hold_reply_after_prior_held_run_requests_blank():
    class LostReplyPort(HoldStagePort):
        async def write(self, data):
            if data.decode().strip() == "SHOTHOLD 0":
                self.commands.append("SHOTHOLD 0")
                return len(data)
            return await super().write(data)

    port = LostReplyPort()
    owner = UnifiedMcu(port, line_timeout_s=0.05)
    await owner.identify()

    async def snap(*args):
        pass

    await owner.software_shots(
        AngularStack(9, 0, 18, 1, -180, 180),
        1,
        [1],
        snap,
        camera_exclusive=True,
        illumination_ready=True,
        bind_stage=False,
        hold_illumination=True,
    )
    with pytest.raises(TimeoutError):
        await owner.software_shots(
            AngularStack(10, 0, 18, 1, -180, 180),
            1,
            [1],
            snap,
            camera_exclusive=True,
            illumination_ready=True,
            bind_stage=False,
        )
    assert port.commands[-2:] == ["SHOTHOLD 0", "00"] and owner.uncertain


@pytest.mark.parametrize("per", [4, 16])
async def test_unbound_shots_skip_motor_preflight_and_preserve_frame_order(per):
    port = OptionalStagePort()
    owner = UnifiedMcu(port)
    await owner.identify()
    observed = []

    async def snap(index, plane, pattern, target):
        observed.append((index, plane, pattern, target))

    await owner.software_shots(
        AngularStack(9, 0, 18, 2, -180, 180),
        per,
        [1] * (2 * per),
        snap,
        camera_exclusive=True,
        illumination_ready=True,
        bind_stage=False,
    )
    assert observed == [(i, i // per, 1, 18 * (i // per)) for i in range(2 * per)]
    assert "SHOTBIND 0" in port.commands
    assert "zasyncxconfig" not in port.commands and "zasyncxstate" not in port.commands
    await owner.software_shots(
        AngularStack(10, 0, 18, 2, -180, 180),
        4,
        [1] * 8,
        snap,
        camera_exclusive=True,
        illumination_ready=True,
    )
    assert "SHOTBIND 1" in port.commands and "zasyncxconfig" in port.commands


async def test_new_firmware_allows_reuse_after_completed_stack():
    class ReusablePort(FakePort):
        async def write(self, data):
            if data == b"identity\n":
                self.commands.append("identity")
                self.emit("ARH-UNIFIED-STACK/0.18")
                return len(data)
            return await super().write(data)

    port = ReusablePort(repeat=True)
    owner = UnifiedMcu(port)
    await owner.identify()
    await owner.stack(AngularStack(7, 0, 18, 3, -180, 180))
    await owner.stack(AngularStack(7, 0, 18, 3, -180, 180))
    assert port.commands.count("START 7") == 2


async def test_only_current_firmware_identity_is_accepted():
    class OldPort(FakePort):
        async def write(self, data):
            if data == b"identity\n":
                self.emit("ARH-UNIFIED-STACK/0.14")
                return len(data)
            return await super().write(data)

    owner = UnifiedMcu(OldPort())
    with pytest.raises(UnifiedMcuError, match="unexpected firmware identity"):
        await owner.identify()


async def test_automatic_camera_prestart_failure_never_sends_start():
    port = ShotFakePort()
    owner = UnifiedMcu(port)
    await owner.identify()

    async def fail():
        assert any(c.startswith("SHOTARM ") for c in port.commands)
        assert not any(c.startswith("SHOTAUTO ") for c in port.commands)
        raise RuntimeError("camera not ready")

    with pytest.raises(RuntimeError, match="camera not ready"):
        await owner.automatic_shots(
            definition(),
            4,
            1,
            50,
            camera_armed=True,
            illumination_ready=True,
            before_start=fail,
        )
    assert not any(c.startswith("SHOTAUTO ") for c in port.commands)
    assert any(c.startswith("SHOTABORT ") for c in port.commands)
    assert owner.uncertain


@pytest.mark.asyncio
@pytest.mark.parametrize("readback", [17, 18, 37])
async def test_bounded_x28_point_move_requires_settled_readback(readback):
    class PointMovePort(FakePort):
        async def write(self, data):
            if data == b"zabs 18\n":
                self.commands.append("zabs 18")
                self.x_position = readback
                self.emit("Z result=0")
                return len(data)
            return await super().write(data)

    port = PointMovePort()
    mcu = UnifiedMcu(port)
    await mcu.identify()
    if readback in (17, 18):
        state = await mcu.move_x_absolute(18, lower=-36, upper=36)
        assert state.position_tenths == state.target_tenths == readback
        assert "zstop" not in port.commands
    else:
        with pytest.raises(UnifiedMcuError, match="outside bounds"):
            await mcu.move_x_absolute(18, lower=-36, upper=36)
        assert port.commands[-1] == "zstop"
        assert mcu.uncertain
    assert port.commands[1:5] == [
        "zasyncxconfig", "zasyncxstate", "zabs 18", "zasyncxstate",
    ]


@pytest.mark.asyncio
async def test_bounded_x28_refuses_out_of_range_start_before_command():
    port = FakePort()
    port.x_position = -37
    mcu = UnifiedMcu(port)
    await mcu.identify()
    with pytest.raises(UnifiedMcuError, match="outside bounds"):
        await mcu.move_x_absolute(18, lower=-36, upper=36)
    assert port.commands == ["identity", "zasyncxconfig", "zasyncxstate"]
