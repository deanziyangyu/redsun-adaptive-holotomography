from __future__ import annotations

from dependency_injector.providers import Dependency

from redsun_aht import providers
from redsun_aht.streams import ALL_STREAMS


def test_stream_names_are_unique_and_namespaced() -> None:
    assert len(ALL_STREAMS) == 5
    assert all(name == "primary" or name.startswith("aht_") for name in ALL_STREAMS)


def test_public_provider_keys_are_typed_dependencies() -> None:
    assert isinstance(providers.DETECTOR_REGISTRY, Dependency)
    assert isinstance(providers.PLAN_REGISTRY, Dependency)
    assert isinstance(providers.CALIBRATION_SELECTION, Dependency)
    assert isinstance(providers.PROCESSING_JOBS, Dependency)
    assert isinstance(providers.DEVICE_PROPERTIES, Dependency)
