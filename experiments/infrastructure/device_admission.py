"""Pure classification for GradFlow's execution-infrastructure contract.

This module deliberately does not probe PyTorch or a driver. Probes are
environment-specific evidence; this function prevents their observations from
being collapsed into the ambiguous word "unavailable".
"""

from __future__ import annotations

from typing import Literal


HostInventory = Literal["present", "absent", "unknown"]
Admission = Literal["passed", "failed", "not_run"]
DeviceStatus = Literal[
    "admitted",
    "visible_unqualified",
    "visible_admission_failed",
    "process_hidden_host_present",
    "host_confirmed_absent",
    "not_visible_host_unknown",
    "probe_failed",
]


def classify_device_admission(
    *,
    process_visible: bool | None,
    host_inventory: HostInventory,
    admission: Admission = "not_run",
) -> DeviceStatus:
    """Classify host presence, process visibility, and numerical admission."""
    if host_inventory not in {"present", "absent", "unknown"}:
        raise ValueError("invalid host inventory status")
    if admission not in {"passed", "failed", "not_run"}:
        raise ValueError("invalid admission status")
    if process_visible is None:
        return "probe_failed"
    if process_visible:
        if admission == "passed":
            return "admitted"
        if admission == "failed":
            return "visible_admission_failed"
        return "visible_unqualified"
    if admission != "not_run":
        raise ValueError("admission cannot run when the device is not visible")
    if host_inventory == "present":
        return "process_hidden_host_present"
    if host_inventory == "absent":
        return "host_confirmed_absent"
    return "not_visible_host_unknown"
