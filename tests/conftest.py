from __future__ import annotations

import pytest

from redsun_aht.configurations import DeviceSelection
from redsun_aht.domain import BackendKind


@pytest.fixture(params=list(BackendKind), ids=lambda backend: backend.value)
def backend_selection(request: pytest.FixtureRequest) -> DeviceSelection:
    """Exercise selection only; even the real case never connects to hardware."""
    backend = request.param
    assert isinstance(backend, BackendKind)
    return DeviceSelection(device_id="detector-a", backend=backend)
