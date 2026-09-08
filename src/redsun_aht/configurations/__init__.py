"""Inspectable build/run factories for AHT composition profiles."""

from .camera_gui import run_camera_acquisition_gui
from .detector_simulation import (
    DualDetectorSimulation,
    build_dual_detector_simulation,
    run_dual_detector_simulation,
)
from .dpct_simulation import (
    DpctSimulation,
    build_dpct_simulation,
    build_panel91_dpct_simulation,
    run_dpct_simulation,
)
from .mcu_test import SimulatedMcuTestSession, build_simulated_mcu_test_session
from .processing_flyer_replay import (
    ProcessingFlyerReplay,
    ProcessingFlyerReplayResult,
    build_processing_flyer_replay,
    processing_flyer_plan,
    run_processing_flyer_replay,
)
from .processing_gui import build_processing_gui_presenter, run_processing_gui
from .processing_simulation import (
    build_processing_container,
    run_processing_simulation,
)
from .profiles import PROFILES, ProfileConfig, load_profile, profile_metadata
from .redsun_simulation import (
    build_redsun_simulation_container,
    run_redsun_simulation_container,
)
from .simulation import SimulationApplication, build_simulation, run_simulation

__all__ = [
    "PROFILES",
    "DpctSimulation",
    "DualDetectorSimulation",
    "ProcessingFlyerReplay",
    "ProcessingFlyerReplayResult",
    "ProfileConfig",
    "SimulatedMcuTestSession",
    "SimulationApplication",
    "build_dpct_simulation",
    "build_dual_detector_simulation",
    "build_panel91_dpct_simulation",
    "build_processing_container",
    "build_processing_flyer_replay",
    "build_processing_gui_presenter",
    "build_redsun_simulation_container",
    "build_simulated_mcu_test_session",
    "build_simulation",
    "load_profile",
    "processing_flyer_plan",
    "profile_metadata",
    "run_camera_acquisition_gui",
    "run_dpct_simulation",
    "run_dual_detector_simulation",
    "run_processing_flyer_replay",
    "run_processing_gui",
    "run_processing_simulation",
    "run_redsun_simulation_container",
    "run_simulation",
]
