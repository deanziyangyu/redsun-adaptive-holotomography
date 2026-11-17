"""Optional exclusive DCAM binding for the ARH desktop session."""

import hashlib
from pathlib import Path

from redsun_aht.device.dcam_process import DcamProcess
from .profile import read_arh_profile


def external_dcam_profile(
    profile_path: Path, camera_config: Path, polarity: str, *, mm_path: Path | None = None,
) -> dict:
    """Prepare an explicit DCAM checkout, leaving hardware closed until a run."""
    profile = read_arh_profile(profile_path)
    if profile.get("mock", True) is not False:
        raise ValueError("External DCAM capture requires an explicitly live profile")
    services = profile["services"]
    mcu = services.get("mcu_service", {})
    mcu_args = mcu.get("args", {})
    if mcu_args.get("zdt-software-only"):
        raise ValueError("External DCAM capture is disabled by the Due profile")
    if mcu_args.get("zdt-stack") is not True:
        raise ValueError("External DCAM capture requires the Due X-stack service")
    revision = mcu_args.get("firmware-revision")
    if not isinstance(revision, str) or not revision.strip() or "REPLACE_" in revision:
        raise ValueError("External DCAM capture requires the installed Due firmware revision")
    if "dcam" in profile.get("devices", {}):
        raise ValueError("Remove the regular DCAM device before taking exclusive DCAM ownership")
    for name, service in services.items():
        if not isinstance(service, dict) or service.get("plugin_id") != "mmcore-camera":
            continue
        args = service.get("args", {})
        if args.get("adapter") == "SpinnakerC" and "mm-config" not in args:
            continue
        source = args.get("mm-config")
        if source:
            rows = Path(source).read_text(encoding="utf-8-sig").splitlines()
            devices = [row.split(",") for row in map(str.strip, rows) if row.startswith("Device,")]
            if len(devices) == 1 and len(devices[0]) == 4 and devices[0][2] == "SpinnakerC":
                continue
        raise ValueError(f"Remove camera service {name!r} before taking exclusive DCAM ownership")

    if polarity not in ("POSITIVE", "NEGATIVE"):
        raise ValueError("Specify the commissioned DCAM polarity")
    camera_config = camera_config.resolve()
    validated_config = camera_config.read_bytes()
    rows = [row.strip() for row in validated_config.decode("utf-8-sig").splitlines()]
    devices = [row.split(",") for row in rows if row.startswith("Device,")]
    if len(devices) != 1 or devices[0][2:] != ["HamamatsuHam", "HamamatsuHam_DCAM"]:
        raise ValueError("Use a dedicated Hamamatsu DCAM configuration")
    label = devices[0][1]
    properties = {}
    core_properties = {"Initialize": {"0", "1"}, "Camera": {label}, "AutoShutter": {"0"}}
    for row in rows:
        fields = row.split(",", 3)
        if not row or row.startswith("#") or row.startswith("Device,"):
            continue
        if len(fields) != 4 or fields[0] != "Property":
            raise ValueError(f"Unsupported directive in dedicated DCAM config: {fields[0]}")
        if fields[1] == "Core":
            if fields[3] not in core_properties.get(fields[2], set()):
                raise ValueError(f"Unsupported Core setting in dedicated DCAM config: {fields[2]}")
        elif fields[1] == label:
            properties[fields[2]] = fields[3]
        else:
            raise ValueError(f"Unexpected property device in dedicated DCAM config: {fields[1]}")
    required = {
        "TRIGGER SOURCE": "EXTERNAL",
        "TRIGGER ACTIVE": "EDGE",
        "TriggerPolarity": polarity,
        "Trigger": "NORMAL",
    }
    for prop, expected in required.items():
        if properties.get(prop) != expected:
            raise ValueError(f"External DCAM config must declare {prop}={expected}")
    storage = Path(profile["storage"]["base_dir"]) / "due_z_stack"
    active_camera = None

    def camera_factory():
        nonlocal active_camera
        if active_camera is not None and not active_camera.released:
            raise RuntimeError("The previous external DCAM checkout did not confirm release")
        if camera_config.read_bytes() != validated_config:
            raise RuntimeError("External DCAM config changed after validation; restart the session")
        active_camera = DcamProcess(
            camera_config, label, polarity, mm_path,
            config_digest=hashlib.sha256(validated_config).hexdigest(),
        )
        return active_camera

    def store_uri(run_id: str) -> str:
        storage.mkdir(parents=True, exist_ok=True)
        return str(storage / f"{run_id}.zarr")

    profile.setdefault("presenters", {}).setdefault("arh_ctrl", {}).update(
        due_capture_factory=camera_factory, due_store_uri=store_uri,
    )
    return profile
