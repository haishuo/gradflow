FC = gfortran
FFLAGS = -O3 -march=native -std=legacy -fallow-argument-mismatch -ffixed-line-length-none
BUILD_DIR := build

.PHONY: all original storage_only original_repaired dynamic regression clean

all: original storage_only original_repaired dynamic

original: $(BUILD_DIR)/weno_original

dynamic: $(BUILD_DIR)/weno_dynamic

storage_only: $(BUILD_DIR)/weno_storage_only

original_repaired: $(BUILD_DIR)/weno_original_repaired

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/weno_original: original_snapshot/weno.f original_snapshot/comm.inc | $(BUILD_DIR)
	$(FC) $(FFLAGS) -I original_snapshot original_snapshot/weno.f -o $@

$(BUILD_DIR)/weno_state.o: dynamic/weno_state.f90 | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J $(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/weno_storage_only.o: dynamic/weno_dynamic.f $(BUILD_DIR)/weno_state.o | $(BUILD_DIR)
	$(FC) $(FFLAGS) -I $(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/weno_storage_only: $(BUILD_DIR)/weno_state.o $(BUILD_DIR)/weno_storage_only.o
	$(FC) $^ -o $@

$(BUILD_DIR)/weno_original_repaired.f: original_snapshot/weno.f dynamic/fixed_form_repairs.patch | $(BUILD_DIR)
	patch --silent --read-only=ignore -o $@ $< < dynamic/fixed_form_repairs.patch

$(BUILD_DIR)/weno_original_repaired: $(BUILD_DIR)/weno_original_repaired.f original_snapshot/comm.inc
	$(FC) $(FFLAGS) -I original_snapshot $< -o $@

$(BUILD_DIR)/weno_dynamic.f: dynamic/weno_dynamic.f dynamic/fixed_form_repairs.patch | $(BUILD_DIR)
	patch --silent --read-only=ignore -o $@ $< < dynamic/fixed_form_repairs.patch

$(BUILD_DIR)/weno_dynamic.o: $(BUILD_DIR)/weno_dynamic.f $(BUILD_DIR)/weno_state.o
	$(FC) $(FFLAGS) -I $(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/weno_dynamic: $(BUILD_DIR)/weno_state.o $(BUILD_DIR)/weno_dynamic.o
	$(FC) $^ -o $@

regression: all
	./check_regression.sh

clean:
	rm -rf -- $(BUILD_DIR)
