FC = gfortran
FFLAGS = -O3 -march=native -std=f2008
PYTHON = python3
BUILD_DIR := build

.PHONY: all fortran3d parity clean

all: fortran3d

fortran3d: $(BUILD_DIR)/shu_euler_3d

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/shu_euler_3d: fortran/shu_euler_3d.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J $(BUILD_DIR) $< -o $@

parity: fortran3d
	$(MAKE) -C ../fortran_scaling dynamic
	$(PYTHON) check_parity.py

clean:
	rm -rf -- $(BUILD_DIR)
