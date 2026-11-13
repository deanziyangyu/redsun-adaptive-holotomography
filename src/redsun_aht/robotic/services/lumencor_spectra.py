"""Serve a dedicated MMCore Lumencor Spectra configuration over PVAccess."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fastcs.attributes import AttrR, AttrRW, AttrW, AttributeIO, AttributeIORef
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, String
from redsun.services import configure_logging

from ._process import controller_id, identity_arguments, run, serve


@dataclass
class LightRef(AttributeIORef):
    field: str = ""


class LightIO(AttributeIO):
    def __init__(self, light):
        super().__init__()
        self.light = light

    async def update(self, attr: AttrR):
        await attr.update(await self.light.call(self.light.read, attr.io_ref.field))

    async def send(self, attr: AttrW, value):
        actual = await self.light.call(self.light.write, attr.io_ref.field, value)
        if isinstance(attr, AttrR):
            await attr.update(actual)


class LumencorController(Controller):
    def __init__(self, core_factory, config, device, shutter, group, presets):
        self.core_factory = core_factory
        self.config = config
        self.device = device
        self.shutter = shutter
        self.group = group
        self.presets = presets
        self.core = None
        self.closing = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumencor-mmcore")
        io = LightIO(self)
        super().__init__(ios=[io])
        self.shutter_open = AttrRW(Bool(), io_ref=LightRef(field="shutter", update_period=0.5))
        self.channel = AttrRW(String(), io_ref=LightRef(field="channel", update_period=0.5))

    async def call(self, operation, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, operation, *args)

    def _load(self):
        self.core = self.core_factory()
        self.core.loadSystemConfiguration(str(self.config))
        loaded = set(self.core.getLoadedDevices())
        if self.shutter not in loaded:
            raise ValueError("MMCore configuration is missing the configured shutter")
        self.core.setAutoShutter(False)
        self._set_shutter(False)
        if self.device not in loaded:
            raise ValueError("MMCore configuration is missing the Lumencor device")
        available = set(self.core.getAvailableConfigs(self.group))
        if not self.presets <= available:
            raise ValueError("MMCore configuration is missing an allowed channel preset")

    def _set_shutter(self, opened):
        self.core.setShutterOpen(self.shutter, opened)
        actual = bool(self.core.getShutterOpen(self.shutter))
        if actual != opened:
            raise RuntimeError("Lumencor shutter readback disagrees with command")
        return actual

    def read(self, field):
        if field == "shutter":
            return bool(self.core.getShutterOpen(self.shutter))
        return str(self.core.getCurrentConfig(self.group))

    def write(self, field, value):
        if self.closing:
            raise RuntimeError("Lumencor service is shutting down")
        if field == "shutter":
            if not value:
                return self._set_shutter(False)
            if self.read("channel") not in self.presets:
                raise RuntimeError("Select an allowed Lumencor channel before opening")
            try:
                return self._set_shutter(True)
            except Exception:
                self._set_shutter(False)
                raise
        if value not in self.presets:
            raise ValueError("Unknown Lumencor channel preset")
        self._set_shutter(False)
        try:
            self.core.setConfig(self.group, value)
            self.core.waitForConfig(self.group, value)
        finally:
            self._set_shutter(False)
        if self.read("channel") != value:
            raise RuntimeError("Lumencor channel readback disagrees with command")
        return value

    def _close(self):
        if self.core is None:
            return
        try:
            if self.shutter in self.core.getLoadedDevices():
                self._set_shutter(False)
        finally:
            self.core.unloadAllDevices()
            self.core = None

    async def connect(self):
        try:
            await self.call(self._load)
            await super().connect()
        except BaseException:
            await self.disconnect()
            raise

    async def disconnect(self):
        if self.closing:
            return
        self.closing = True
        try:
            await super().disconnect()
        finally:
            try:
                await self.call(self._close)
            finally:
                self.executor.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmcore-config", type=Path, required=True)
    parser.add_argument("--mm-path")
    parser.add_argument("--device-label", required=True)
    parser.add_argument("--shutter-device", required=True)
    parser.add_argument("--channel-group", required=True)
    parser.add_argument("--allowed-presets", required=True, help="Comma-separated MMCore preset names")
    identity_arguments(parser, "lumencor-spectra")
    options = parser.parse_args(argv)
    presets = {value.strip() for value in options.allowed_presets.split(",") if value.strip()}
    if not options.mmcore_config.is_file() or not presets or any(
        "REPLACE_" in value for value in (
            str(options.mmcore_config), options.mm_path or "", options.device_label,
            options.shutter_device, options.channel_group, *presets,
        )
    ):
        parser.error("Supply a dedicated MMCore configuration and configured device and channel names")
    from pymmcore_plus import CMMCorePlus

    def make_core():
        return CMMCorePlus(mm_path=options.mm_path) if options.mm_path else CMMCorePlus()

    configure_logging()
    controller = LumencorController(
        make_core, options.mmcore_config, options.device_label,
        options.shutter_device, options.channel_group, presets,
    )
    async def run_service():
        try:
            await serve(controller, controller_id(options))
        finally:
            await controller.disconnect()

    return run(run_service())


if __name__ == "__main__":
    raise SystemExit(main())
