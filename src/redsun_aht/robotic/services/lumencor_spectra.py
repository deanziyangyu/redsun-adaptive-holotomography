"""Serve a dedicated MMCore Lumencor Spectra configuration over PVAccess."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from fastcs.attributes import AttrR, AttrRW, AttrW, AttributeIO, AttributeIORef
from fastcs.controllers import Controller
from fastcs.datatypes import Bool, Int, String
from redsun.services import configure_logging

from ._process import controller_id, identity_arguments, run, serve


@dataclass
class LightRef(AttributeIORef):
    field: str = ""


class LightIO(AttributeIO[bool | int | str, LightRef]):
    def __init__(self, light):
        super().__init__()
        self.light = light
        self._lock = asyncio.Lock()

    async def update(self, attr: AttrR):
        async with self._lock:
            await attr.update(await self.light.call(self.light.read, attr.io_ref.field))

    async def send(self, attr: AttrW, value):
        field = attr.io_ref.field
        async with self._lock:
            actual = await self.light.call(self.light.write, field, value)
            if field == "channel":
                # A successful channel write verifies that the shutter is closed.
                await self.light.shutter_open.update(False)
                await self.light.intensity.update(await self.light.call(self.light.read, "intensity"))
            if isinstance(attr, AttrR):
                await attr.update(actual)


class LumencorController(Controller):
    def __init__(self, core_factory, config, device, shutter, group, presets, *, brightfield_is_off=False):
        self.core_factory = core_factory
        self.config = config
        self.device = device
        self.shutter = shutter
        self.group = group
        self.presets = presets
        self.brightfield_is_off = brightfield_is_off
        self.core = None
        self.closing = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumencor-mmcore")
        io = LightIO(self)
        super().__init__(ios=[io])
        self.shutter_open = AttrRW(Bool(), io_ref=LightRef(field="shutter", update_period=0.5))
        self.intensity = AttrRW(Int(), io_ref=LightRef(field="intensity", update_period=0.5))
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

    def _intensity_property(self):
        color = self.read("channel").partition("_")[2]
        if color not in {"blue", "cyan", "green", "red", "teal", "violet"}:
            return None
        return f"{color.title()}_Level"

    def _intensity_limits(self, prop):
        return (
            self.core.getPropertyLowerLimit(self.device, prop),
            self.core.getPropertyUpperLimit(self.device, prop),
        )

    def read(self, field):
        if field == "intensity":
            prop = self._intensity_property()
            if prop is None:
                return 0
            low, high = self._intensity_limits(prop)
            level = float(self.core.getProperty(self.device, prop))
            return round(100 * (level - low) / (high - low))
        if field == "shutter":
            return bool(self.core.getShutterOpen(self.shutter))
        current = str(self.core.getCurrentConfig(self.group))
        if current == "brightfield" and self.brightfield_is_off:
            return "off"
        return current

    def write(self, field, value):
        if self.closing:
            raise RuntimeError("Lumencor service is shutting down")
        if field == "shutter":
            if not value:
                return self._set_shutter(False)
            channel = self.read("channel")
            if channel == "off" or channel not in self.presets:
                raise RuntimeError("Select an illumination preset before opening the Lumencor shutter")
            try:
                return self._set_shutter(True)
            except Exception:
                self._set_shutter(False)
                raise
        if field == "intensity":
            if type(value) is not int or not 0 <= value <= 100:
                raise ValueError("Spectra intensity must be an integer from 0 to 100 percent")
            prop = self._intensity_property()
            if prop is None:
                raise RuntimeError("Select a Spectra channel before setting intensity")
            low, high = self._intensity_limits(prop)
            level = round(low + (high - low) * value / 100)
            self.core.setProperty(self.device, prop, str(level))
            if float(self.core.getProperty(self.device, prop)) != level:
                raise RuntimeError("Spectra intensity readback disagrees with command")
            return round(100 * (level - low) / (high - low))
        if value not in self.presets:
            raise ValueError("Unknown Lumencor channel preset")
        self._set_shutter(False)
        try:
            self.core.setConfig(self.group, value)
            self.core.waitForConfig(self.group, value)
        finally:
            self._set_shutter(False)
        expected = "off" if value == "brightfield" and self.brightfield_is_off else value
        if self.read("channel") != expected:
            raise RuntimeError("Lumencor channel readback disagrees with command")
        return expected

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
            await self.shutter_open.update(await self.call(self.read, "shutter"))
            await self.channel.update(await self.call(self.read, "channel"))
            await self.intensity.update(await self.call(self.read, "intensity"))
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
    parser.add_argument("--brightfield-is-off", action="store_true")
    identity_arguments(parser, "lumencor-spectra")
    options = parser.parse_args(argv)
    presets = {value.strip() for value in options.allowed_presets.split(",") if value.strip()}
    if options.brightfield_is_off and not {"off", "brightfield"} <= presets:
        parser.error("--brightfield-is-off requires both presets")
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
        brightfield_is_off=options.brightfield_is_off,
    )
    async def run_service():
        try:
            await serve(controller, controller_id(options))
        finally:
            await controller.disconnect()

    return run(run_service())


if __name__ == "__main__":
    raise SystemExit(main())
