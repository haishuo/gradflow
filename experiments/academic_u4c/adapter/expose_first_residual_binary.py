#!/usr/bin/env python3
"""Export the first generated OpenSBLI residual as an interior float64 file."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    source = args.source
    text = source.read_text()
    anchor = "\nint iteration_range_8_block0[] = {0, block0np0};"
    if text.count(anchor) != 1:
        raise RuntimeError("generated RK-update anchor was not unique")

    instrumentation = r'''
// U4-C instrumentation: retrieve the first native residual before RK.
int u4c_disp[OPS_MAX_DIM] = {0};
int u4c_size[OPS_MAX_DIM] = {0};
ops_dat_get_extents(Residual0_B0, 0, u4c_disp, u4c_size);
double *u4c_buffer = (double*)malloc(sizeof(double)*u4c_size[0]);
ops_dat_fetch_data(Residual0_B0, 0, (char*)u4c_buffer);
const char *u4c_path = getenv("U4C_RHS_PATH");
if (u4c_path == NULL) {
  ops_printf("U4C_RHS_PATH is required\n");
  free(u4c_buffer);
  ops_exit();
  return 2;
}
FILE *u4c_file = fopen(u4c_path, "wb");
if (u4c_file == NULL) {
  ops_printf("Unable to open U4C_RHS_PATH\n");
  free(u4c_buffer);
  ops_exit();
  return 3;
}
const int u4c_offset = -u4c_disp[0];
const size_t u4c_written = fwrite(
  u4c_buffer + u4c_offset, sizeof(double), block0np0, u4c_file
);
fclose(u4c_file);
free(u4c_buffer);
if (u4c_written != (size_t)block0np0) {
  ops_printf("Incomplete U4-C residual write\n");
  ops_exit();
  return 4;
}
ops_exit();
return 0;
'''
    text = text.replace(anchor, "\n" + instrumentation + anchor)
    source.write_text(text)


if __name__ == "__main__":
    main()
