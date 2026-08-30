#!/usr/bin/env python3
"""Turn a generated U4-B application into the frozen U4-C benchmark driver."""

from __future__ import annotations

import argparse
from pathlib import Path


INITIAL_OLD = (
    "phi_B0(0) = 0.4 + 0.1*cos(6*M_PI*x0_B0(0)) + "
    "sin(2*M_PI*x0_B0(0));"
)
INITIAL_NEW = (
    "phi_B0(0) = 0.4 + sin(74*M_PI*x0_B0(0)) + "
    "0.1*cos(182*M_PI*x0_B0(0));"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("kernel", type=Path)
    args = parser.parse_args()

    kernel = args.kernel.read_text()
    if kernel.count(INITIAL_OLD) != 1:
        raise RuntimeError("generated initialization expression was not unique")
    args.kernel.write_text(kernel.replace(INITIAL_OLD, INITIAL_NEW))

    source = args.source.read_text()
    include_anchor = '#include "io.h"\n'
    if source.count(include_anchor) != 1:
        raise RuntimeError("generated include anchor was not unique")
    source = source.replace(
        include_anchor,
        include_anchor
        + "#include <chrono>\n"
        + "#include <stdio.h>\n"
        + "#ifdef U4C_CUDA\n#include <cuda_runtime.h>\n#endif\n",
    )

    start = source.index("// Initialize loop timers")
    output = (
        "HDF5_IO_Write_0_opensbliblock00(opensbliblock00, phi_B0, "
        "Residual0_B0, x0_B0, HDF5_timing);"
    )
    finish = source.index(output, start) + len(output)
    body = r'''// U4-C frozen generated-operator driver.
const char *u4c_mode_value = getenv("U4C_MODE");
const char *u4c_output_path = getenv("U4C_RHS_PATH");
const char *u4c_state_path = getenv("U4C_STATE_PATH");
if (u4c_state_path == NULL) {
  ops_printf("U4C_STATE_PATH is required\n");
  ops_exit();
  return 2;
}
FILE *u4c_state_file = fopen(u4c_state_path, "rb");
if (u4c_state_file == NULL) {
  ops_printf("Unable to open U4C_STATE_PATH\n");
  ops_exit();
  return 2;
}
double *u4c_state_interior = (double*)malloc(sizeof(double)*block0np0);
const size_t u4c_state_read = fread(
  u4c_state_interior, sizeof(double), block0np0, u4c_state_file);
fclose(u4c_state_file);
if (u4c_state_read != (size_t)block0np0) {
  ops_printf("Incomplete U4-C state read\n");
  free(u4c_state_interior);
  ops_exit();
  return 2;
}
int u4c_state_range[] = {0, block0np0};
ops_dat_set_data_slab_memspace(
  phi_B0, 0, (char*)u4c_state_interior, u4c_state_range, OPS_HOST);
const int u4c_warmups = getenv("U4C_WARMUPS") ? atoi(getenv("U4C_WARMUPS")) : 5;
const int u4c_samples = getenv("U4C_SAMPLES") ? atoi(getenv("U4C_SAMPLES")) : 20;
const bool u4c_qualification = u4c_mode_value != NULL && strcmp(u4c_mode_value, "qualify") == 0;
const bool u4c_launch = u4c_mode_value != NULL && strcmp(u4c_mode_value, "launch") == 0;
const bool u4c_transfer = u4c_mode_value != NULL && strcmp(u4c_mode_value, "transfer") == 0;
const int u4c_total = (u4c_qualification || u4c_launch) ? 1 : u4c_warmups + u4c_samples;
int u4c_reconstruction_range[] = {-1, block0np0 + 1};
int u4c_residual_range[] = {0, block0np0};
double *u4c_transfer_buffer = u4c_transfer
  ? (double*)malloc(sizeof(double)*block0np0) : NULL;
#ifdef U4C_CUDA
cudaEvent_t u4c_start_event, u4c_end_event;
cudaEventCreate(&u4c_start_event);
cudaEventCreate(&u4c_end_event);
#endif
for (int u4c_repetition = 0; u4c_repetition < u4c_total; ++u4c_repetition) {
  std::chrono::steady_clock::time_point u4c_started;
  if (u4c_transfer) {
    u4c_started = std::chrono::steady_clock::now();
    ops_dat_set_data_slab_memspace(
      phi_B0, 0, (char*)u4c_state_interior, u4c_state_range, OPS_HOST);
  }
  ops_halo_transfer(periodicBC_direction0_side0_3_block0);
  ops_halo_transfer(periodicBC_direction0_side1_4_block0);
#ifdef U4C_CUDA
  if (!u4c_transfer) cudaEventRecord(u4c_start_event);
#else
  if (!u4c_transfer) u4c_started = std::chrono::steady_clock::now();
#endif
  ops_par_loop(opensbliblock00Kernel000, "LFWeno_reconstruction_0_direction",
    opensbliblock00, 1, u4c_reconstruction_range,
    ops_arg_dat(phi_B0, 1, stencil_0_23_1, "double", OPS_READ),
    ops_arg_dat(wk0_B0, 1, stencil_0_00_1, "double", OPS_WRITE));
  ops_par_loop(opensbliblock00Kernel002, "LFWeno Residual", opensbliblock00,
    1, u4c_residual_range,
    ops_arg_dat(wk0_B0, 1, stencil_0_10_1, "double", OPS_READ),
    ops_arg_dat(Residual0_B0, 1, stencil_0_00_1, "double", OPS_WRITE));
  double u4c_elapsed_ms = 0.0;
  if (u4c_transfer) {
    ops_dat_fetch_data(Residual0_B0, 0, (char*)u4c_transfer_buffer);
    u4c_elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - u4c_started).count();
  } else {
#ifdef U4C_CUDA
    cudaEventRecord(u4c_end_event);
    cudaEventSynchronize(u4c_end_event);
    float u4c_event_ms = 0.0f;
    cudaEventElapsedTime(&u4c_event_ms, u4c_start_event, u4c_end_event);
    u4c_elapsed_ms = (double)u4c_event_ms;
#else
    u4c_elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - u4c_started).count();
#endif
  }
  if (!u4c_qualification && !u4c_launch && u4c_repetition >= u4c_warmups) {
    ops_printf("U4C_SAMPLE %.17g\n", u4c_elapsed_ms);
  }
}
#ifdef U4C_CUDA
cudaEventDestroy(u4c_start_event);
cudaEventDestroy(u4c_end_event);
#endif
free(u4c_transfer_buffer);
free(u4c_state_interior);
int u4c_disp[OPS_MAX_DIM] = {0};
int u4c_size[OPS_MAX_DIM] = {0};
ops_dat_get_extents(Residual0_B0, 0, u4c_disp, u4c_size);
double *u4c_buffer = (double*)malloc(sizeof(double)*u4c_size[0]);
ops_dat_fetch_data(Residual0_B0, 0, (char*)u4c_buffer);
if (u4c_output_path != NULL) {
  FILE *u4c_file = fopen(u4c_output_path, "wb");
  if (u4c_file == NULL) {
    ops_printf("Unable to open U4C_RHS_PATH\n");
    free(u4c_buffer);
    ops_exit();
    return 3;
  }
  const int u4c_offset = -u4c_disp[0];
  const size_t u4c_written = fwrite(
    u4c_buffer + u4c_offset, sizeof(double), block0np0, u4c_file);
  fclose(u4c_file);
  if (u4c_written != (size_t)block0np0) {
    ops_printf("Incomplete U4-C residual write\n");
    free(u4c_buffer);
    ops_exit();
    return 4;
  }
}
double u4c_checksum = 0.0;
const int u4c_offset = -u4c_disp[0];
for (int u4c_j = 0; u4c_j < block0np0; ++u4c_j) {
  u4c_checksum += u4c_buffer[u4c_offset + u4c_j];
}
ops_printf("U4C_CHECKSUM %.17g\n", u4c_checksum);
free(u4c_buffer);
ops_exit();
return 0;'''
    args.source.write_text(source[:start] + body + source[finish:])


if __name__ == "__main__":
    main()
