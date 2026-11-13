"""Presenter admits only one camera and one MCU device."""

from unittest.mock import Mock

from redsun_aht.device.unassociated_stack_session import UnassociatedStackPresenter


def test_plan_requires_unambiguous_device_pair():
    device = Mock()
    for cameras, stacks in (
        ({}, {}),
        ({"camera": device}, {}),
        ({}, {"mcu": device}),
        ({"a": device, "b": device}, {"mcu": device}),
    ):
        assert not UnassociatedStackPresenter(
            "test", cameras=cameras, stacks=stacks
        ).plan_map()
    presenter = UnassociatedStackPresenter(
        "test", cameras={"camera": device}, stacks={"mcu": device}
    )
    assert "unassociated_mcu_stack" in presenter.plan_map()
