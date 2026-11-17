"""Check desktop travel-limit readbacks without constructing a Qt application."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from psygnal import Signal
from redsun.session._wiring import Wiring

from redsun_aht.configurations.arh import ArhSession, _read_rotary_configuration, _read_stage_limits, run_arh
from redsun_aht.presenter.arh import ArhPresenter
from redsun_aht.presenter.detector import DetectorPresenter
from redsun_aht.robotic.device.asi._service import AsiSerialStage
from redsun_aht.robotic.services.lumencor_spectra import LumencorController
from redsun_aht.view.arh import ArhControls
from redsun_aht.view.detector import DetectorView
from redsun_aht.view.image import ImageView
from redsun_aht.contracts import Roi


class TestArhDesktopReadback(unittest.IsolatedAsyncioTestCase):
    def test_direct_pva_restart_uses_fresh_context_and_restores_original(self):
        import os

        from redsun_aht.configurations import arh_pva

        backend = arh_pva._p4p
        original = Mock()
        clients = [Mock(), Mock()]
        factory = backend.context
        transport = arh_pva.TRANSPORTS["pv-access"]
        key = "EPICS_PVA_NAME_SERVERS"
        with patch.dict(os.environ, {key: "original:5075"}), patch.object(backend, "_context", original):
            with patch.object(backend, "Context", side_effect=clients) as create:
                for port, client in zip((63001, 63002), clients, strict=True):
                    with arh_pva.arh_direct_pva():
                        direct = arh_pva.TRANSPORTS["pv-access"]
                        direct._ports["mcu_service"] = port
                        direct.publish("mcu_service")
                        self.assertIn(f"127.0.0.1:{port}", os.environ[key])
                        self.assertIs(backend.context(), client)
                        self.assertIs(backend.context(), client)
                    client.close.assert_called_once_with()
                    self.assertIs(backend.context, factory)
                    self.assertIs(backend._context, original)
                    self.assertIs(arh_pva.TRANSPORTS["pv-access"], transport)
                    self.assertEqual(os.environ[key], "original:5075")
                self.assertEqual(create.call_count, 2)
        original.close.assert_not_called()

    def test_session_injects_arh_acquisition_setup(self):
        from redsun import AsPresenter, Session
        from tempfile import TemporaryDirectory
        from typing import ClassVar
        from redsun_aht.presenter.acquisition import ArhAcquisitionPresenter

        class HeadlessSession(Session):
            config: ClassVar[dict[str, object]] = {
                "session": "arh-planner-injection", "strict": True, "mock": True,
                "presenters": {"det_ctrl": {}, "arh_ctrl": {"include_workcell": True}, "acq_ctrl": {}},
            }
            det_ctrl: AsPresenter[DetectorPresenter]
            arh_ctrl: AsPresenter[ArhPresenter]
            acq_ctrl: AsPresenter[ArhAcquisitionPresenter]

        with TemporaryDirectory() as directory:
            session = HeadlessSession({"storage": {"base_dir": directory}})
            try:
                session.build()
                self.assertIs(session.acq_ctrl._arh, session.arh_ctrl)
                self.assertIn("live_stream", session.acq_ctrl.plans)
            finally:
                session.shutdown()

    def test_acquisition_planner_collects_workcell_plans_from_profile(self):
        from redsun_aht.presenter.acquisition import AcquisitionPresenter, ArhAcquisitionPresenter
        from redsun_aht.robotic.device.workcell_webcam import WorkcellWebcam

        config = {
            "services": {"webcam_service": {"plugin_id": "workcell-webcam"}},
            "devices": {"webcam": {"plugin_id": "workcell-webcam", "service": "webcam_service"}},
        }
        module = "redsun_aht.configurations.arh"
        with patch(f"{module}.read_arh_profile", return_value=config), patch(f"{module}.ArhSession"):
            run_arh(["--profile", "fixture.yaml"])
        devices = {"webcam": WorkcellWebcam("TEST:", name="webcam")}
        arh = ArhPresenter("arh_ctrl", **config["presenters"]["arh_ctrl"])
        detector = DetectorPresenter("det_ctrl", detectors={})
        planner = ArhAcquisitionPresenter("acq_ctrl", devices=devices)
        try:
            planner.setup({"arh_ctrl": arh, "det_ctrl": detector}, {}, Mock())
            for name in ("workcell_snap", "workcell_record"):
                parameter = next(p for p in planner.plans[name].parameters if p.name == "webcam")
                self.assertEqual(tuple(parameter.choices), ("webcam",))
            stream = next(p for p in planner.plans["live_stream"].parameters if p.name == "webcam")
            self.assertEqual(tuple(stream.choices), ("webcam",))
            with patch.object(arh, "workcell_preview", return_value=iter(())) as preview:
                list(planner.live_stream((), 3, False, webcam=(devices["webcam"],)))
                preview.assert_called_once_with(devices["webcam"], 3, continuous=False)
            holospim = AcquisitionPresenter("acq_ctrl", devices={})
            holospim.setup({"det_ctrl": detector}, {}, Mock())
            self.assertNotIn("webcam", {p.name for p in holospim.plans["live_stream"].parameters})
            with patch.object(detector, "live_stream", return_value=iter(())) as stream:
                list(holospim.live_stream((), 3, False))
                stream.assert_called_once_with((), 3, False, holospim.live_stream.__func__.__defaults__[-1])
        finally:
            detector.shutdown()

    def test_hardware_widget_constructor_satisfies_attachment_contract(self):
        from redsun.session._protocols import AttachableComponent
        from redsun_aht.view import arh

        widget = SimpleNamespace(
            placement=ArhControls.placement, setObjectName=Mock(),
            confirm_rotary_zero=Mock(), _load_led_map=Mock(),
            _confirm_dynamic_dpct=Mock(),
        )
        widget.set_led_array_capability = ArhControls.set_led_array_capability.__get__(widget)
        fake_super = lambda: SimpleNamespace(__init__=Mock())
        with patch.object(arh, "QtW", Mock()), patch.object(arh, "super", fake_super, create=True):
            ArhControls.__init__(widget, "hardware_widget", Mock())
        self.assertIsInstance(widget, AttachableComponent)
        self.assertEqual(widget.name, "hardware_widget")
        widget.set_led_array_capability(61, True)
        widget._led_array.setEnabled.assert_called_with(True)
        widget._led_pixel.setRange.assert_called_with(0, 60)
        widget._led_map_button.setEnabled.assert_called_with(True)

    def test_napari_grid_action_injects_embedded_viewer_until_close(self):
        from napari._app_model import get_app_model
        from napari._qt import qt_event_loop, qt_viewer
        from napari.components import ViewerModel
        from napari.utils.action_manager import action_manager
        from redsun_aht import hooks

        # Supply a stand-in model and widget without creating a Qt application.
        viewer = SimpleNamespace(
            canvas=SimpleNamespace(grid=SimpleNamespace(enabled=False, stride=1)),
            layers=[],
        )
        hook = hooks.NapariApplication()
        store = get_app_model().injection_store
        before = store.provide(ViewerModel)
        with (
            patch.object(qt_event_loop, "get_qapp", return_value=object()),
            patch.object(qt_viewer, "QtViewer", return_value=Mock()),
            patch("napari.components.ViewerModel", Mock(return_value=viewer)) as model_factory,
            patch.object(store, "register_provider", wraps=store.register_provider) as register,
        ):
            register.side_effect = lambda provider, _: store.register(
                providers={ViewerModel: provider}
            )
            try:
                hook.create_application([])
                hook.create_application([])
                model_factory.assert_called_once()
                register.assert_called_once()
                action_manager.trigger("napari:toggle_grid")
                self.assertTrue(viewer.canvas.grid.enabled)
                action_manager.trigger("napari:toggle_grid")
                self.assertFalse(viewer.canvas.grid.enabled)
            finally:
                hook.close()
        self.assertIs(store.provide(ViewerModel), before)

    def test_shutdown_accepts_viewer_already_destroyed_by_qt_parent(self):
        from redsun_aht import hooks

        hook = hooks.NapariApplication()
        hook._viewer = Mock()
        hook._qt_viewer = Mock()
        hook._qt_viewer.close.side_effect = RuntimeError(
            "wrapped C/C++ object of type QtViewer has been deleted"
        )
        with patch.object(hooks, "_APPLICATION", hook):
            hook.shutdown()
            self.assertIsNone(hooks.current_application())
        hook.shutdown()

    def test_due_plan_registry_requires_matching_client_and_service(self):
        for binding, stack_enabled in (("due", True), ("due", False), ("missing", True), (None, True)):
            with self.subTest(binding=binding, stack_enabled=stack_enabled):
                config = {
                    "services": {"due": {"plugin_id": "unified-emmsh",
                                         "args": {"zdt-stack": stack_enabled,
                                                  "zdt-software": True, "software-dpct": True}}},
                    "devices": {"mcu": {"plugin_id": "unified-emmsh", "service": binding}}
                    if binding else {},
                }
                module = "redsun_aht.configurations.arh"
                with patch(f"{module}.read_arh_profile", return_value=config), patch(f"{module}.ArhSession"):
                    run_arh(["--profile", "fixture.yaml"])
                options = config["presenters"]["arh_ctrl"]
                plans = ArhPresenter("arh_ctrl", **options).plan_map()
                for name in ("emmsh_stop", "emmsh_rotary_target", "arh_dpct"):
                    self.assertEqual(name in plans, binding == "due", name)
                for name in ("due_z_stack_status", "due_z_stack_abort", "due_software_z_stack"):
                    self.assertEqual(name in plans, binding == "due" and stack_enabled, name)
                self.assertIn("lx4000_z_stack", plans)

    async def test_stopped_startup_keeps_commissioned_ranges_for_the_panel(self):
        axes = ("roll", "pitch", "yaw")
        targets = dict.fromkeys(axes, 8)
        readback = AsyncMock(return_value=json.dumps({
            "rotary_limits": dict.fromkeys(axes, [2, 12]), "steps": targets,
        }))
        mcu = SimpleNamespace(
            rotary_enabled=SimpleNamespace(get_value=AsyncMock(return_value=False)),
            commanded_steps=SimpleNamespace(get_value=readback),
        )
        self.assertEqual(await _read_rotary_configuration(mcu),
                         (dict.fromkeys(axes, (2, 12)), targets, False))
        readback.return_value = json.dumps({"rotary_limits": {}, "steps": targets})
        self.assertEqual(await _read_rotary_configuration(mcu), ({}, {}, False))

    async def test_partial_rotary_profile_enables_only_configured_axes(self):
        axes = ("roll", "pitch", "yaw")
        mcu = SimpleNamespace(
            rotary_enabled=SimpleNamespace(get_value=AsyncMock(return_value=True)),
            commanded_steps=SimpleNamespace(get_value=AsyncMock(return_value=json.dumps({
                "rotary_limits": {"pitch": [0, 4], "yaw": [0, 4]},
                "steps": dict.fromkeys(axes, 0),
            }))),
        )
        bounds, _, enabled = await _read_rotary_configuration(mcu)
        self.assertEqual(bounds, {"pitch": (0, 4), "yaw": (0, 4)})
        self.assertTrue(enabled)

    def test_roi_overlay_editability_tracks_selected_camera_lock(self):
        dcam = SimpleNamespace(editable=True, visible=True)
        flir = SimpleNamespace(editable=True, visible=True)
        view = SimpleNamespace(_roi_layers={"dcam": dcam, "camera": flir})

        ImageView.set_roi_locked(view, frozenset({"dcam"}))
        self.assertFalse(dcam.editable)
        self.assertTrue(dcam.visible)
        self.assertTrue(flir.editable)

        ImageView.set_roi_locked(view, frozenset())
        self.assertTrue(dcam.editable)

    async def test_deferred_roi_checks_camera_lock_when_applied(self):
        signal = SimpleNamespace(
            name="dcam-roi", set=AsyncMock(),
            read=AsyncMock(return_value={"dcam-roi": {"value": "0,0,2,2"}}),
        )
        deferrals = Mock()
        presenter = SimpleNamespace(
            _resolve_setting=lambda name: ("dcam", "roi", signal),
            _locked_detectors=frozenset(), _deferrals=deferrals,
            _rois={}, logger=Mock(), sig_new_configuration=Mock(),
        )
        await DetectorPresenter.set(presenter, "dcam-roi", "0,0,2,2")
        apply = deferrals.request.call_args.args[0]
        DetectorPresenter.set_locked(presenter, frozenset({"dcam"}))
        await apply()
        signal.set.assert_not_awaited()
        presenter.sig_new_configuration.emit.assert_not_called()

        DetectorPresenter.set_locked(presenter, frozenset())
        await DetectorPresenter.set(presenter, "dcam-roi", "0,0,2,2")
        await deferrals.request.call_args.args[0]()
        signal.set.assert_awaited_once_with("0,0,2,2")

    def test_image_roi_drag_cannot_update_locked_editor(self):
        panel = Mock()
        panel.isEnabled.return_value = False
        view = SimpleNamespace(settings_controls={"dcam": SimpleNamespace(roi_panel=panel)})
        roi = Roi(0, 0, 2, 2)
        DetectorView.on_roi_drawn(view, "dcam", roi)
        panel.draw.assert_not_called()
        panel.isEnabled.return_value = True
        DetectorView.on_roi_drawn(view, "dcam", roi)
        panel.draw.assert_called_once_with(roi)

    def test_acquisition_holds_stage_across_engine_lock_changes(self):
        class HeadlessSession:
            _set_positioner_locks = ArhSession._set_positioner_locks
            hold = ArhSession._hold_positioner
            release = ArhSession._release_positioner

        session = HeadlessSession()
        session._positioner_locks = frozenset()
        session._positioner_plan = None
        session._plan_cameras = frozenset()
        session._plan_stages = frozenset()
        session.devices = {"sample_stage": AsiSerialStage("TEST:", name="sample_stage")}
        session.positioner_ctrl = Mock()
        session.positioner_widget = Mock()
        session.det_widget = Mock()
        session.det_ctrl = Mock()
        session.img_widget = Mock()
        session.hold("live_stream", {"detectors": ["camera", "dcam"]})
        session._set_positioner_locks(frozenset())
        session.det_widget.set_locked.assert_called_with(frozenset({"camera", "dcam"}))
        session.positioner_ctrl.set_locked.assert_called_with(frozenset())
        session.release("live_stream")
        session.det_widget.set_locked.assert_called_with(frozenset())

        session.hold("lx4000_z_stack", {"camera": "camera"})
        session.det_widget.set_locked.assert_called_with(frozenset({"sample_stage", "camera"}))
        session.img_widget.set_roi_locked.assert_called_with(frozenset({"sample_stage", "camera"}))
        session._set_positioner_locks(frozenset({"dcam"}))
        session.release("filter_wheel_slot", "Invalid slot")
        session.positioner_ctrl.set_locked.assert_called_with(frozenset({"sample_stage", "dcam"}))
        session.release("lx4000_z_stack")
        session.positioner_ctrl.set_locked.assert_called_with(frozenset({"dcam"}))
        session.positioner_widget.set_locked.assert_called_with(frozenset({"dcam"}))
        session.det_widget.set_locked.assert_called_with(frozenset({"dcam"}))

        for plan in ("due_software_z_stack", "arh_dpct", "arh_hand_eye_3d"):
            with self.subTest(plan=plan):
                session._set_positioner_locks(frozenset())
                session.hold(plan, {"camera": "dcam"})
                session.positioner_ctrl.set_locked.assert_called_with(frozenset({"sample_stage"}))
                session.img_widget.set_roi_locked.assert_called_with(frozenset({"sample_stage", "dcam"}))
                session.release(plan, "Injected acquisition failure")
                session.positioner_ctrl.set_locked.assert_called_with(frozenset())
                session.img_widget.set_roi_locked.assert_called_with(frozenset())

    def test_stop_unlocks_hardware_panel_after_active_plan_finishes(self):
        panel = SimpleNamespace(
            _active_plan=None, _plan=None, _status=Mock(), setEnabled=Mock(),
        )
        panel.plan_finished = lambda name: ArhControls.plan_finished(panel, name)
        ArhControls.plan_started(panel, "due_z_stack", {})
        ArhControls.plan_stopped(panel, "due_z_stack")
        self.assertEqual([call.args for call in panel.setEnabled.call_args_list], [(False,), (True,)])
        self.assertIsNone(panel._active_plan)

    def test_hardware_request_uses_active_plan_registry(self):
        class HeadlessSession:
            sig_launch = Signal(str, dict)
            dispatch = ArhSession._launch_hardware_plan

        session = HeadlessSession()
        session._hardware_names = {"illumination": "spectra"}
        session.acq_ctrl = SimpleNamespace(
            plans={"spectra_shutter": object()}, launch=Mock(), accepting_plan=True,
        )
        session.hardware_widget = SimpleNamespace(plan_failed=Mock())
        wiring = Wiring({}, {}, {}, frontend=Mock())
        wiring.link(session.sig_launch, session.dispatch)
        values = {"light": "illumination", "opened": False}
        session.sig_launch.emit("spectra_shutter", values)
        session.sig_launch.emit("filter_wheel_slot", {"slot": 4})
        session.acq_ctrl.launch.assert_called_once_with("spectra_shutter", {**values, "light": "spectra"})
        session.hardware_widget.plan_failed.assert_called_once_with(
            "filter_wheel_slot", "filter_wheel_slot is unavailable in this profile",
        )
        session.acq_ctrl.accepting_plan = False
        session.sig_launch.emit("spectra_shutter", values)
        session.acq_ctrl.launch.assert_called_once_with("spectra_shutter", {**values, "light": "spectra"})
        session.hardware_widget.plan_failed.assert_called_with(
            "spectra_shutter", "An acquisition is already active",
        )

    def test_hardware_panel_ignores_second_request_until_first_finishes(self):
        panel = SimpleNamespace(
            _plan=None, _active_plan=None, _status=Mock(), sig_launch=Mock(),
        )
        ArhControls.launch(panel, "spectra_shutter", light="illumination", opened=False)
        ArhControls.launch(panel, "filter_wheel_slot", wheel="wheel", slot=4)
        panel.sig_launch.emit.assert_called_once_with(
            "spectra_shutter", {"light": "illumination", "opened": False},
        )
        self.assertEqual(panel._plan, "spectra_shutter")

    def test_rejected_wheel_command_does_not_unlock_active_acquisition(self):
        panel = SimpleNamespace(
            _active_plan=None, _plan=None, _status=Mock(), setEnabled=Mock(),
        )
        panel.plan_finished = lambda name: ArhControls.plan_finished(panel, name)

        ArhControls.plan_started(panel, "arh_dpct", {})
        ArhControls.plan_failed(panel, "filter_wheel_slot", "Invalid slot")
        panel.setEnabled.assert_called_once_with(False)
        self.assertEqual(panel._active_plan, "arh_dpct")

        ArhControls.plan_finished(panel, "arh_dpct")
        panel.setEnabled.assert_called_with(True)
        self.assertIsNone(panel._active_plan)

    def test_brightfield_alias_keeps_shutter_closed(self):
        core = Mock()
        controller = LumencorController(
            lambda: core, None, "Spectra", "Spectra", "spectracolors",
            {"off", "brightfield", "640_red"}, brightfield_is_off=True,
        )
        self.addCleanup(controller.executor.shutdown)
        controller.core = core
        view = SimpleNamespace(_open_shutter=Mock(), _illumination_state=Mock(), _set_intensity=Mock())
        ArhControls.set_illumination_readback(view, "brightfield", False)
        view._open_shutter.setEnabled.assert_called_with(False)

        for preset, allowed in (("brightfield", False), ("640_red", True)):
            with self.subTest(preset=preset):
                core.getCurrentConfig.return_value = preset
                channel = controller.read("channel")
                ArhControls.set_illumination_readback(view, channel, False)
                view._open_shutter.setEnabled.assert_called_with(allowed)
                if not allowed:
                    self.assertEqual(channel, "off")
                    with self.assertRaisesRegex(RuntimeError, "Select an illumination preset"):
                        controller.write("shutter", True)
                    core.setShutterOpen.assert_not_called()

    async def test_travel_limits_require_three_ordered_finite_intervals(self):
        xy = AsyncMock(return_value='{"x": [-1000, 1000], "y": [-500, 1500]}')
        z = AsyncMock(return_value="[-1000, 1000]")
        stage = SimpleNamespace(motion=SimpleNamespace(
            xy_limits=SimpleNamespace(get_value=xy),
            z_limits=SimpleNamespace(get_value=z),
        ))
        self.assertEqual(await _read_stage_limits(stage), {
            "x": [-1000, 1000], "y": [-500, 1500], "z": [-1000, 1000],
        })
        xy.assert_awaited_once_with(cached=False)
        z.assert_awaited_once_with(cached=False)

        for invalid in ("null", "[0, 0]", "[false, 1000]", "[0, Infinity]"):
            with self.subTest(z=invalid), self.assertRaisesRegex(ValueError, "Invalid LX4000 Z"):
                z.return_value = invalid
                await _read_stage_limits(stage)

        xy.return_value = '{"x": [-1000, 1000]}'
        with self.assertRaisesRegex(ValueError, "X and Y travel intervals"):
            await _read_stage_limits(stage)
