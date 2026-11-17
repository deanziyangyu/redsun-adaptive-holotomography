"""Direct TCP discovery for Redsun-managed ARH services on Windows."""

from contextlib import contextmanager
import os
import socket

from ophyd_async.epics.core import _p4p
from redsun.services._transports import PVAccess, TRANSPORTS, add_to_env


class DirectPVA(PVAccess):
    """Assign each subprocess a loopback port and publish its TCP address."""

    def __init__(self):
        self._ports: dict[str, int] = {}
        for service, variable in (
            ("lx4000_service", "ARH_LX4000_PVA_PORT"),
            ("webcam_service", "ARH_WEBCAM_PVA_PORT"),
        ):
            port = os.environ.get(variable)
            if port is None:
                continue
            if not port.isascii() or not port.isdecimal() or not 1024 <= int(port) <= 65535:
                raise ValueError(f"{variable} must be a TCP port from 1024 to 65535")
            if int(port) in self._ports.values():
                raise ValueError(f"{variable} must use a distinct TCP port")
            self._ports[service] = int(port)

    def reserve(self, service: str) -> dict[str, str]:
        if service not in self._ports:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            while port in self._ports.values():
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
            self._ports[service] = port
        port = str(self._ports[service])
        return {
            "EPICS_PVAS_INTF_ADDR_LIST": "127.0.0.1",
            "EPICS_PVAS_SERVER_PORT": port,
            "EPICS_PVA_NAME_SERVERS": f"127.0.0.1:{port}",
        }

    def publish(self, service: str) -> None:
        add_to_env("EPICS_PVA_NAME_SERVERS", f"127.0.0.1:{self._ports[service]}")


@contextmanager
def arh_direct_pva():
    """Use TCP discovery for the lifetime of a standalone ARH session."""
    previous = TRANSPORTS["pv-access"]
    previous_context = _p4p._context
    previous_factory = _p4p.context
    key = "EPICS_PVA_NAME_SERVERS"
    original = os.environ.get(key)
    TRANSPORTS["pv-access"] = DirectPVA()
    # The ophyd-async client reads name servers at construction. A later
    # standalone session has different ports even if it uses the same PVs.
    _p4p._context = None
    def session_context():
        if _p4p._context is None:
            _p4p._context = _p4p.Context("pva", nt=False)
        return _p4p._context

    _p4p.context = session_context
    try:
        yield
    finally:
        context = _p4p._context
        _p4p._context = previous_context
        _p4p.context = previous_factory
        TRANSPORTS["pv-access"] = previous
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
        if context is not None:
            context.close()
