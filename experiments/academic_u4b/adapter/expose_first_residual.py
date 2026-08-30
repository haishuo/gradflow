#!/usr/bin/env python3
"""Expose OpenSBLI's first generated residual before its RK update."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    source = args.source
    text = source.read_text()
    output_calls = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("HDF5_IO_Write_0_opensbliblock00(")
    ]
    if len(output_calls) != 1:
        raise RuntimeError("expected exactly one generated final HDF5 output call")

    anchor = "\nint iteration_range_8_block0[] = {0, block0np0};"
    if text.count(anchor) != 1:
        raise RuntimeError("generated RK-update anchor was not unique")
    insertion = (
        "\n// U4-B instrumentation: export the native residual before any RK update.\n"
        + output_calls[0]
        + "\nops_exit();\nreturn 0;\n"
        + anchor
    )
    source.write_text(text.replace(anchor, insertion))


if __name__ == "__main__":
    main()
