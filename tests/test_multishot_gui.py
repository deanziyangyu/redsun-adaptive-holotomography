from __future__ import annotations

import time

import pytest

from redsun_aht.view.multishot import (
    MultishotScanRequest,
    MultishotScanSummary,
    MultishotScanWidget,
)


class _Client:
    def __init__(self) -> None:
        self.request: MultishotScanRequest | None = None

    def run(self, request: MultishotScanRequest) -> MultishotScanSummary:
        self.request = request
        return MultishotScanSummary("gui-run", request.shot_count, "a" * 64, 12.5, "8")


def test_request_rejects_a_scan_past_the_admitted_z_envelope() -> None:
    with pytest.raises(ValueError, match="100 um"):
        MultishotScanRequest(z_positions=102, z_step_um=1)


def test_widget_runs_a_bounded_four_shot_request() -> None:
    from qtpy import QtWidgets

    client = _Client()
    application_instance = QtWidgets.QApplication.instance()
    application = (
        application_instance
        if isinstance(application_instance, QtWidgets.QApplication)
        else QtWidgets.QApplication([])
    )
    widget = MultishotScanWidget(client)
    widget.run_scan()
    deadline = time.monotonic() + 2
    while (
        "Completed gui-run" not in widget.status_label.text()
        and time.monotonic() < deadline
    ):
        application.processEvents()
        time.sleep(0.01)

    assert client.request is not None
    assert client.request.shot_count == 160
    assert "Completed gui-run" in widget.status_label.text()
