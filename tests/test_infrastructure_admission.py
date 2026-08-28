from __future__ import annotations

import pytest

from experiments.infrastructure.device_admission import classify_device_admission


@pytest.mark.parametrize(
    ("process_visible", "host_inventory", "admission", "expected"),
    (
        (True, "present", "passed", "admitted"),
        (True, "present", "failed", "visible_admission_failed"),
        (True, "unknown", "not_run", "visible_unqualified"),
        (False, "present", "not_run", "process_hidden_host_present"),
        (False, "absent", "not_run", "host_confirmed_absent"),
        (False, "unknown", "not_run", "not_visible_host_unknown"),
        (None, "unknown", "not_run", "probe_failed"),
    ),
)
def test_infrastructure_admission_statuses(
    process_visible: bool | None,
    host_inventory: str,
    admission: str,
    expected: str,
) -> None:
    assert classify_device_admission(
        process_visible=process_visible,
        host_inventory=host_inventory,  # type: ignore[arg-type]
        admission=admission,  # type: ignore[arg-type]
    ) == expected


def test_admission_cannot_pass_when_device_is_hidden() -> None:
    with pytest.raises(ValueError, match="cannot run"):
        classify_device_admission(
            process_visible=False,
            host_inventory="present",
            admission="passed",
        )
