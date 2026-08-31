#!/usr/bin/env python3
"""Offline semantic and checksum verifier for the Moody A4 replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TAG_COMMIT = "c5e8ab81ef5b33a2138b2db33afc538398b6f57f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    record = json.loads((evidence / "second_machine.json").read_text())

    assert record["schema"] == "gradflow-academic-a4-second-machine-v1"
    assert record["status"] in {
        "pass",
        "pass_with_limitations",
        "fail_needs_investigation",
    }
    assert record["source_commit"] == TAG_COMMIT
    assert record["environment"]["git_commit"] == TAG_COMMIT
    assert not record["environment"]["git_status"]
    assert record["environment"]["hostname"] != "forge"
    assert record["environment"]["execution_context"] == "standalone"
    assert record["workspace_contract"] == "/mnt/projects"
    assert len(record["sentinels"]) == 6
    assert len(record["a2_workers"]) == 36
    assert set(record["qualification"]) == {
        "sentinels_passed",
        "a1_completed",
        "a3_agreement_passed",
        "a2_worker_surface_complete",
        "a2_graph_contract_passed",
        "binary32_cuda_materially_useful",
        "admission_failures",
    }
    if record["status"] == "pass":
        assert all(
            record["qualification"][key]
            for key in (
                "sentinels_passed",
                "a1_completed",
                "a3_agreement_passed",
                "a2_worker_surface_complete",
                "a2_graph_contract_passed",
                "binary32_cuda_materially_useful",
            )
        )
        assert not record["qualification"]["admission_failures"]

    expected = {}
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    assert expected
    for relative, digest in expected.items():
        path = evidence / relative
        assert path.is_file(), relative
        assert sha256(path) == digest, relative
    print(
        f"Moody second-machine evidence verified: status={record['status']}, "
        f"files={len(expected)}"
    )


if __name__ == "__main__":
    main()
