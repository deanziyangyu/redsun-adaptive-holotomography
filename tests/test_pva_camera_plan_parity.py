"""The same software DPCT plan over both camera service contracts."""

import asyncio
import uuid

import numpy as np
import pytest
from bluesky import RunEngine
from fastcs.transports.epics.pva.transport import EpicsPVATransport
from mmcore_fakes import DATA_KEY, LABEL, FakeCameraCore, core_of
from redsun.path_provider import SessionPathProvider
from redsun.services import serve_controller
from test_dpct_plan import FakeIllumination

from redsun_aht.contracts import PlanStatus
from redsun_aht.device.mmcore.camera import MMCamera
from redsun_aht.device.spinnaker_camera import SpinnakerCamera
from redsun_aht.plans.dpct_software import software_dpct_plan
from redsun_aht.plans.live import live_stream, prepare_live_stream
from redsun_aht.services.mmcore_camera import MMCameraController
from redsun_aht.services.spinnaker_camera import SpinnakerCameraController


@pytest.mark.parametrize("direct", [False, True], ids=["mmcore", "spinnaker"])
async def test_software_dpct_through_pva_camera(tmp_path, direct):
    class CameraSource(FakeCameraCore):
        def isBufferOverflowed(self):
            return False

        def popNextImageAndMD(self):
            frame, metadata = super().popNextImageAndMD()
            if direct:
                metadata = {
                    "frame_id": str(self.popped - 1),
                    "timestamp_ns": str(self.popped * 1000),
                    "raw_pixel_format": "Mono12p",
                    "converted_pixel_format": "Mono16",
                }
            return frame, metadata

        def close(self):
            pass

    source = CameraSource(pixel_type="16bit", exposure_limits=(0.1, 1000.0))
    controller_type = SpinnakerCameraController if direct else MMCameraController
    controller = controller_type(
        core_of(source), LABEL, DATA_KEY, defer_initial_frame=True
    )
    prefix = "DP" + uuid.uuid4().hex[:10].upper()
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    camera_type = SpinnakerCamera if direct else MMCamera
    camera = camera_type(
        prefix + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="parity"),
        name="camera",
    )

    async def produce():
        while True:
            if source.sequencing:
                source.produce()
            await asyncio.sleep(0.01)

    producer = asyncio.create_task(produce())
    try:
        await asyncio.wait_for(camera.connect(), 10)

        def run():
            engine = RunEngine({}, context_managers=[])
            documents = []
            engine.subscribe(lambda name, doc: documents.append((name, doc)))
            outcomes = []

            def plan():
                outcomes.append(
                    (
                        yield from software_dpct_plan(
                            [camera],
                            None,
                            FakeIllumination(),
                            run_id="parity",
                            settle_s=0,
                        )
                    )
                )

            engine(plan())
            return outcomes, documents

        outcomes, documents = await asyncio.wait_for(asyncio.to_thread(run), 30)
        assert outcomes[0].status is PlanStatus.COMPLETE
        assert len([doc for name, doc in documents if name == "event"]) == 4
    finally:
        producer.cancel()
        server.cancel()
        await asyncio.gather(producer, server, return_exceptions=True)


async def test_spinnaker_live_roi_and_exposure_changes(tmp_path):
    class CameraSource(FakeCameraCore):
        def produce(self, frames=1):
            for _ in range(frames):
                self._buffer.put(np.ones((self.roi[3], self.roi[2]), dtype=self.dtype))

        def stopSequenceAcquisition(self):
            super().stopSequenceAcquisition()
            while not self._buffer.empty():
                self._buffer.get_nowait()

        def close(self):
            pass

    source = CameraSource(pixel_type="16bit", exposure_limits=(0.1, 1000))
    controller = SpinnakerCameraController(
        core_of(source), LABEL, DATA_KEY, defer_initial_frame=True
    )
    prefix = "SL" + uuid.uuid4().hex[:10].upper()
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    camera = SpinnakerCamera(
        prefix + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="live"),
        name="camera",
    )

    async def produce():
        while True:
            if source.sequencing:
                source.produce()
            await asyncio.sleep(0.01)

    producer = asyncio.create_task(produce())
    try:
        await asyncio.wait_for(camera.connect(), 10)
        await camera.acquire.set(True)
        await camera.capture_frame()
        await camera.roi.set("0,0,3,2")
        await camera.exposure.set(20.0)
        for _ in range(10):
            frame = await asyncio.wait_for(camera.capture_frame(), 5)
            if frame.shape == (2, 3):
                break
        assert frame.shape == (2, 3)
        assert frame.exposure_s == pytest.approx(0.02)
        assert await camera.roi.get_value(cached=False) == "0,0,3,2"
        assert source.sequencing
        await camera.acquire.set(False)
        await camera.file_path.set(str(tmp_path / "open-capture"))
        await camera.num_capture.set(4)
        await camera.capture.set(True)
        with pytest.raises(Exception, match="idle capture"):
            await camera.roi.set("0,0,2,2")
        assert source.roi == (0, 0, 3, 2)
        await camera.capture.set(False)
    finally:
        producer.cancel()
        server.cancel()
        await asyncio.gather(producer, server, return_exceptions=True)


@pytest.mark.parametrize("count", [0, 8], ids=["unbounded", "finite"])
async def test_spinnaker_live_stream_after_timed_capture(tmp_path, count):
    from bluesky import plan_stubs as bps
    from ophyd_async.core import TriggerInfo

    class CameraSource(FakeCameraCore):
        def extra_frames_are_errors(self):
            return False

        def popNextImageAndMD(self):
            frame = self.popNextImage()
            return frame, {"frame_id": str(self.popped - 1), "timestamp_ns": "42"}

        def close(self):
            pass

    source = CameraSource(pixel_type="16bit")
    controller = SpinnakerCameraController(
        core_of(source),
        LABEL,
        DATA_KEY,
        defer_initial_frame=True,
        buffered_capture=True,
        finite_capture=True,
    )
    prefix = "SU" + uuid.uuid4().hex[:10].upper()
    server = asyncio.create_task(
        serve_controller(controller, EpicsPVATransport(), prefix)
    )
    camera = SpinnakerCamera(
        prefix + ":",
        path_provider=SessionPathProvider(base_dir=tmp_path, session="live"),
        name="camera",
    )

    async def produce():
        while True:
            if source.sequencing:
                source.produce()
            await asyncio.sleep(0.01)

    producer = asyncio.create_task(produce())
    try:
        await asyncio.wait_for(camera.connect(), 10)
        await camera.record_frame_timing.set(True)

        def run():
            engine = RunEngine({}, context_managers=[])
            docs = []
            engine.subscribe(lambda name, doc: docs.append((name, doc)))

            def plan():
                yield from bps.stage(camera)
                yield from prepare_live_stream(
                    [camera],
                    TriggerInfo(number_of_events=count),
                    "frames",
                    declare=False,
                )
                yield from live_stream(
                    [camera], "frames", until=bps.sleep(0.3), flush_period_s=0.1
                )

            engine(plan())
            return docs

        documents = await asyncio.wait_for(asyncio.to_thread(run), 30)
        assert any(name == "stream_resource" for name, _ in documents)
        assert any(name == "stream_datum" for name, _ in documents)
        assert source.popped > 0
        assert await camera.record_frame_timing.get_value(cached=False) == (count != 0)
        assert not await camera.acquire.get_value(cached=False)
    finally:
        producer.cancel()
        server.cancel()
        await asyncio.gather(producer, server, return_exceptions=True)
