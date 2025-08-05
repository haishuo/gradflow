"""
GradFlow Reference Data Loader

Loads canonical reference data for bit-perfect validation against FORTRAN WENO.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch

from .. import DATA_ROOT


class WENOReference:
    """Loads and manages WENO reference data for validation"""

    def __init__(self, reference_dir: Optional[Path] = None):
        """
        Initialize reference data loader

        Args:
            reference_dir: Path to reference data. If None, uses default location.
        """
        if reference_dir is None:
            # Check multiple possible locations for reference data
            possible_locations = [
                Path("/mnt/projects/weno-reference/reference/sod_shock_working"),
                Path("/mnt/projects/weno-reference/reference"),
                Path(__file__).parent.parent.parent
                / "reference-fortran"
                / "sod_shock_working",
                Path(__file__).parent.parent.parent / "reference-fortran",
                DATA_ROOT / "validation" / "canonical_reference",
            ]

            # Find the first location that exists
            for loc in possible_locations:
                if loc.exists():
                    reference_dir = loc
                    break

            if reference_dir is None:
                raise FileNotFoundError(
                    f"Could not find reference data. Searched in: {possible_locations\
    }"
                )

        self.reference_dir = Path(reference_dir)
        self._data = {}
        self._metadata = {}

        # Load metadata if available
        metadata_file = self.reference_dir / "reference_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self._metadata = json.load(f)

    def load_sod_shock_reference(self, format: str = "fortran") -> Dict[str, Any]:
        """
        Load Sod shock tube reference data

        Args:
            format: 'hdf5', 'numpy', or 'fortran'

        Returns:
            Dictionary containing reference data
        """
        if "sod_shock" in self._data:
            return self._data["sod_shock"]

        data = {}

        if format == "hdf5":
            data = self._load_hdf5()
        elif format == "numpy":
            data = self._load_numpy()
        elif format == "fortran":
            data = self._load_fortran_files()
        else:
            raise ValueError(f"Unknown format: {format}")

        self._data["sod_shock"] = data
        return data

    def _load_hdf5(self) -> Dict[str, Any]:
        """Load from HDF5 format (recommended for precision)"""
        h5_file = self.reference_dir / "sod_shock_reference.h5"

        if not h5_file.exists():
            # If HDF5 doesn't exist, try to load from FORTRAN files directly
            return self._load_fortran_files()

        data = {}

        with h5py.File(h5_file, "r") as f:
            # Load raw FORTRAN data
            data["fort8"] = {
                "data": f["fort8/data"][:],
                "description": f["fort8"].attrs["description"],
            }

            data["fort9"] = {
                "data": f["fort9/data"][:],
                "description": f["fort9"].attrs["description"],
            }

            # Load structured grid
            grid_info = {}
            for key in f["structured/grid"].attrs:
                grid_info[key] = f["structured/grid"].attrs[key]
            for key in f["structured/grid"].keys():
                grid_info[key] = f["structured/grid"][key][:]

            fields = {}
            for field_name in f["structured/fields"].keys():
                fields[field_name] = f["structured/fields"][field_name][:]

            data["structured"] = {"grid": grid_info, "fields": fields}

            # Load validation points
            validation = {
                "shock_location": f["validation_points"].attrs["shock_location"]
            }

            for state_name in ["left_state", "right_state"]:
                state_data = {}
                for key in f[f"validation_points/{state_name}"].attrs:
                    state_data[key] = f[f"validation_points/{state_name}"].attrs[key]\
    
                validation[state_name] = state_data

            ratios = {}
            for key in f["validation_points/expected_ratios"].attrs:
                ratios[key] = f["validation_points/expected_ratios"].attrs[key]
            validation["expected_ratios"] = ratios

            data["validation_points"] = validation

        return data

    def _load_numpy(self) -> Dict[str, Any]:
        """Load from NumPy format"""
        npz_file = self.reference_dir / "sod_shock_reference.npz"

        if not npz_file.exists():
            # If NPZ doesn't exist, try FORTRAN files
            return self._load_fortran_files()

        npz_data = np.load(npz_file)

        data = {
            "fort8": {"data": npz_data["fort8_data"]},
            "fort9": {"data": npz_data["fort9_data"]},
            "structured": {
                "grid": {
                    "x_coords": npz_data["x_coords"],
                    "y_coords": npz_data["y_coords"],
                },
                "fields": {
                    "density": npz_data["density_2d"],
                    "pressure": npz_data["pressure_2d"],
                    "u_velocity": npz_data["u_velocity_2d"],
                    "v_velocity": npz_data["v_velocity_2d"],
                },
            },
        }

        return data

    def _load_fortran_files(self) -> Dict[str, Any]:
        """Load directly from FORTRAN output files"""
        # Try different possible names for the FORTRAN output files
        fort8_candidates = [
            self.reference_dir / "sod_shock_reference_1d.dat",
            self.reference_dir / "fort.8",
            self.reference_dir / "fort8.dat",
        ]

        fort9_candidates = [
            self.reference_dir / "sod_shock_reference_2d.dat",
            self.reference_dir / "fort.9",
            self.reference_dir / "fort9.dat",
        ]

        fort8_file = None
        fort9_file = None

        for candidate in fort8_candidates:
            if candidate.exists():
                fort8_file = candidate
                break

        for candidate in fort9_candidates:
            if candidate.exists():
                fort9_file = candidate
                break

        if not fort8_file or not fort9_file:
            available_files = list(self.reference_dir.glob("*"))
            raise FileNotFoundError(
                f"FORTRAN output files not found in {self.reference_dir}. "
                f"Available files: {[f.name for f in available_files[:10]]}"
            )

        fort8_data = np.loadtxt(fort8_file)

        # Load fort.9 - it includes boundary points, so it's (nx+1)*(ny+1) = 21*11 = \
    231
        with open(fort9_file, "r") as f:
            lines = f.readlines()

        # Skip any header lines
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("zone") and not line.startswith("ZONE"):
                try:
                    # Try to parse as floats
                    values = list(map(float, line.split()))
                    if len(values) == 6:  # x, y, rho, u, v, p
                        data_lines.append(values)
                except ValueError:
                    continue

        fort9_data = np.array(data_lines)

        # The grid is actually 21x11 (includes boundaries)
        nx_with_boundary = 21
        ny_with_boundary = 11
        total_points = nx_with_boundary * ny_with_boundary

        if len(fort9_data) != total_points:
            raise ValueError(
                f"Expected {total_points} points in fort.9, got {len(fort9_data)}. "
                f"This suggests nx={nx_with_boundary}, ny={ny_with_boundary} with bou\
    ndaries."
            )

        # Extract the interior points (remove boundaries)
        # FORTRAN outputs with boundaries, so actual grid is 20x10
        nx = 20
        ny = 10

        # Reshape including boundaries
        full_data = fort9_data.reshape(ny_with_boundary, nx_with_boundary, 6)

        # Extract interior points (skip first and last in each dimension)
        interior_data = full_data[0:ny, 0:nx, :]

        # Extract fields
        x_coords = interior_data[0, :, 0]
        y_coords = interior_data[:, 0, 1]
        density = interior_data[:, :, 2]
        u_velocity = interior_data[:, :, 3]
        v_velocity = interior_data[:, :, 4]
        pressure = interior_data[:, :, 5]

        data = {
            "fort8": {"data": fort8_data},
            "fort9": {"data": fort9_data},
            "structured": {
                "grid": {
                    "x_coords": x_coords,
                    "y_coords": y_coords,
                    "nx": nx,
                    "ny": ny,
                },
                "fields": {
                    "density": density,
                    "pressure": pressure,
                    "u_velocity": u_velocity,
                    "v_velocity": v_velocity,
                },
            },
            "validation_points": {
                "shock_location": 5.0,
                "left_state": {
                    "density": 1.0,
                    "pressure": 1.0,
                    "u_velocity": 0.0,
                    "v_velocity": 0.0,
                },
                "right_state": {
                    "density": 0.125,
                    "pressure": 0.1,
                    "u_velocity": 0.0,
                    "v_velocity": 0.0,
                },
                "expected_ratios": {
                    "density_ratio": 8.0,
                    "pressure_ratio": 10.0,
                },
            },
        }

        return data

    def get_validation_points(self) -> Dict[str, Any]:
        """Get key validation points for quick testing"""
        data = self.load_sod_shock_reference()
        return data.get("validation_points", {})

    def get_structured_field(self, field_name: str) -> np.ndarray:
        """
        Get a structured field (density, pressure, etc.)

        Args:
            field_name: 'density', 'pressure', 'u_velocity', 'v_velocity'

        Returns:
            2D array of field values [ny, nx]
        """
        data = self.load_sod_shock_reference()
        return data["structured"]["fields"][field_name]

    def get_grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get grid coordinates"""
        data = self.load_sod_shock_reference()
        grid = data["structured"]["grid"]
        return grid["x_coords"], grid["y_coords"]

    def to_torch(
        self,
        field_name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """
        Convert reference data to PyTorch tensor

        Args:
            field_name: Field to convert
            device: PyTorch device
            dtype: PyTorch dtype (use float64 for bit-perfect matching)

        Returns:
            PyTorch tensor
        """
        field_data = self.get_structured_field(field_name)
        return torch.tensor(field_data, device=device, dtype=dtype)

    def compare_solution(
        self,
        gradflow_result: np.ndarray,
        field_name: str,
        tolerance: float = 1e-12,
    ) -> Dict[str, Any]:
        """
        Compare GradFlow solution against reference

        Args:
            gradflow_result: GradFlow computed solution (numpy array)
            field_name: Field being compared
            tolerance: Numerical tolerance

        Returns:
            Comparison results
        """
        # Ensure we have numpy array
        if hasattr(gradflow_result, "detach"):
            gradflow_result = gradflow_result.detach().cpu().numpy()
        elif hasattr(gradflow_result, "numpy"):
            gradflow_result = gradflow_result.numpy()

        reference_data = self.get_structured_field(field_name)

        # Ensure shapes match
        if gradflow_result.shape != reference_data.shape:
            return {
                "passed": False,
                "error": (
                    f"Shape mismatch: {gradflow_result.shape} vs "
                    f"{reference_data.shape}"
                ),
            }

        # Compute differences
        absolute_diff = np.abs(gradflow_result - reference_data)
        relative_diff = np.abs(absolute_diff / (reference_data + 1e-15))

        max_abs_error = np.max(absolute_diff)
        max_rel_error = np.max(relative_diff)
        rms_error = np.sqrt(np.mean(absolute_diff**2))

        passed = max_abs_error < tolerance

        return {
            "passed": passed,
            "max_absolute_error": max_abs_error,
            "max_relative_error": max_rel_error,
            "rms_error": rms_error,
            "tolerance": tolerance,
            "field_name": field_name,
            "shape": gradflow_result.shape,
        }

    def quick_validation_test(self) -> Dict[str, Any]:
        """Run quick validation test on key points"""
        validation_points = self.get_validation_points()

        if not validation_points:
            return {"status": "No validation points available"}

        left = validation_points["left_state"]
        right = validation_points["right_state"]
        expected = validation_points["expected_ratios"]

        # Check density ratio
        actual_density_ratio = left["density"] / right["density"]
        actual_pressure_ratio = left["pressure"] / right["pressure"]

        density_error = abs(actual_density_ratio - expected["density_ratio"])
        pressure_error = abs(actual_pressure_ratio - expected["pressure_ratio"])

        return {
            "status": "success",
            "left_state": left,
            "right_state": right,
            "actual_ratios": {
                "density": actual_density_ratio,
                "pressure": actual_pressure_ratio,
            },
            "expected_ratios": expected,
            "errors": {
                "density_ratio": density_error,
                "pressure_ratio": pressure_error,
            },
            "shock_location": validation_points["shock_location"],
        }

    def print_summary(self):
        """Print summary of available reference data"""
        print("📊 WENO Reference Data Summary")
        print("=" * 40)

        if self._metadata:
            print(f"Description: {self._metadata.get('description', 'N/A')}")
            print(f"Source: {self._metadata.get('source', 'N/A')}")
            print(f"Problem: {self._metadata.get('problem', 'N/A')}")

        print(f"Reference directory: {self.reference_dir}")

        # List available files
        print("\nAvailable files:")
        for file in self.reference_dir.glob("*"):
            print(f"  📁 {file.name}")

        # Quick validation test
        validation = self.quick_validation_test()
        if validation.get("status") == "success":
            print("\n✅ Quick validation test:")
            print(f"   Shock location: {validation['shock_location']}")
            left_density = validation["left_state"]["density"]
            right_density = validation["right_state"]["density"]
            print(f"   Left state density: {left_density:.6f}")
            print(f"   Right state density: {right_density:.6f}")
            actual_density = validation["actual_ratios"]["density"]
            expected_density = validation["expected_ratios"]["density"]
            print(
                f"   Density ratio: {actual_density:.2f} "
                f"(expected: {expected_density:.2f})"
            )
            actual_pressure = validation["actual_ratios"]["pressure"]
            expected_pressure = validation["expected_ratios"]["pressure"]
            print(
                f"   Pressure ratio: {actual_pressure:.2f} "
                f"(expected: {expected_pressure:.2f})"
            )


# Convenience functions
def load_sod_reference(format: str = "fortran") -> WENOReference:
    """Load Sod shock tube reference data"""
    # Use the actual Sod shock working directory
    ref_dir = Path("/mnt/projects/weno-reference/reference/sod_shock_working")
    if not ref_dir.exists():
        # Fall back to symlink
        ref_dir = (
            Path(__file__).parent.parent.parent
            / "reference-fortran"
            / "sod_shock_working"
        )

    ref = WENOReference(reference_dir=ref_dir)
    ref.load_sod_shock_reference(format=format)
    return ref


def validate_against_reference(
    gradflow_result: np.ndarray, field_name: str, tolerance: float = 1e-12
) -> Dict[str, Any]:
    """Quick validation against reference data"""
    ref = WENOReference()
    return ref.compare_solution(gradflow_result, field_name, tolerance)
