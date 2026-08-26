"""Verified caller-owned tensor adapters for DVEB portable ABIs.

The ABI is intentionally narrower than GradFlow's PyTorch implementation: it
accepts one contiguous CPU float32 3-D Euler state and a positive fixed step
count.  The compiled pipeline may then execute on CPU SIMD/OpenMP or CUDA.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import Tensor


ABI_VERSION = 1
DEVICE_ABI_VERSION = 2
_OK = 0
_TARGETS = {"auto": 0, "cpu": 1, "cuda": 2}
_TARGET_NAMES = {0: "auto", 1: "cpu", 2: "cuda"}


class DvebAbiError(RuntimeError):
    """A stable refusal or execution error returned by portable ABI v1."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"DVEB ABI v1 status {status}: {message}")
        self.status = status
        self.message = message


class _Query(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("scalar_type", ctypes.c_uint32),
        ("dimensions", ctypes.c_uint32),
        ("components", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("required_elements", ctypes.c_uint64),
        ("program_sha256", ctypes.c_char * 65),
        ("module_sha256", ctypes.c_char * 65),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _Request(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("intervals", ctypes.c_int32),
        ("steps", ctypes.c_int32),
        ("target", ctypes.c_uint32),
        ("cpu_workers", ctypes.c_int32),
        ("endpoint", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("input", ctypes.POINTER(ctypes.c_float)),
        ("input_count", ctypes.c_uint64),
        ("output", ctypes.POINTER(ctypes.c_float)),
        ("output_capacity", ctypes.c_uint64),
        ("model_path", ctypes.c_char_p),
        ("verified_model_sha256", ctypes.c_char_p),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _Result(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("selected_target", ctypes.c_uint32),
        ("selected_cpu_workers", ctypes.c_int32),
        ("finite", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("output_count", ctypes.c_uint64),
        ("execution_seconds", ctypes.c_double),
        ("total_seconds", ctypes.c_double),
        ("peak_bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _DeviceCreateRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("intervals", ctypes.c_int32),
        ("device_ordinal", ctypes.c_int32),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _DeviceCreateResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("reserved0", ctypes.c_uint32),
        ("required_elements", ctypes.c_uint64),
        ("workspace_bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _DeviceRunRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("steps", ctypes.c_int32),
        ("reserved0", ctypes.c_uint32),
        ("input_device", ctypes.POINTER(ctypes.c_float)),
        ("input_count", ctypes.c_uint64),
        ("output_device", ctypes.POINTER(ctypes.c_float)),
        ("output_capacity", ctypes.c_uint64),
        ("cuda_stream", ctypes.c_void_p),
        ("reserved", ctypes.c_uint64 * 4),
    ]


class _DeviceRunResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_ordinal", ctypes.c_int32),
        ("synchronized", ctypes.c_uint32),
        ("output_count", ctypes.c_uint64),
        ("execution_seconds", ctypes.c_double),
        ("total_seconds", ctypes.c_double),
        ("workspace_bytes", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64 * 4),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DvebArtifact:
    """Hash-pinned ABI library plus optional verified placement model."""

    library: Path
    library_sha256: str
    program_sha256: str
    module_sha256: str
    device_library: Path | None = None
    device_library_sha256: str | None = None
    model: Path | None = None
    verified_model_sha256: str | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | os.PathLike[str],
        *,
        model: str | os.PathLike[str] | None = None,
        verified_model_sha256: str | None = None,
    ) -> "DvebArtifact":
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        if manifest.get("schema") not in {"dveb-artifact-v2", "dveb-artifact-v3"}:
            raise ValueError("unsupported DVEB artifact manifest schema")
        abi = manifest.get("abi", {})
        if abi.get("version") != ABI_VERSION:
            raise ValueError("DVEB artifact does not provide portable ABI v1")
        library = Path(abi["library"])
        if not library.is_absolute():
            library = path.parent / library
        if not library.is_file() or _sha256(library) != abi.get("library_sha256"):
            raise ValueError("DVEB ABI library is missing or differs from its manifest")
        header = Path(abi["header"])
        if not header.is_absolute():
            header = path.parent / header
        if not header.is_file() or _sha256(header) != abi.get("header_sha256"):
            raise ValueError("DVEB ABI header is missing or differs from its manifest")
        device_library = None
        device_library_sha256 = None
        device_abi = manifest.get("device_abi")
        if device_abi is not None:
            if device_abi.get("version") != DEVICE_ABI_VERSION:
                raise ValueError("DVEB artifact has an unsupported device ABI")
            device_library = Path(device_abi["library"])
            if not device_library.is_absolute():
                device_library = path.parent / device_library
            if (not device_library.is_file()
                    or _sha256(device_library) != device_abi.get("library_sha256")):
                raise ValueError(
                    "DVEB device ABI library is missing or differs from its manifest"
                )
            device_header = Path(device_abi["header"])
            if not device_header.is_absolute():
                device_header = path.parent / device_header
            if (not device_header.is_file()
                    or _sha256(device_header) != device_abi.get("header_sha256")):
                raise ValueError(
                    "DVEB device ABI header is missing or differs from its manifest"
                )
            device_library_sha256 = device_abi["library_sha256"]
        artifact = cls(
            library=library,
            library_sha256=abi["library_sha256"],
            program_sha256=manifest["program_sha256"],
            module_sha256=manifest["module_sha256"],
            device_library=device_library,
            device_library_sha256=device_library_sha256,
            model=Path(model).resolve() if model is not None else None,
            verified_model_sha256=verified_model_sha256,
        )
        artifact._validate_model_record()
        return artifact

    @classmethod
    def from_environment(cls) -> "DvebArtifact | None":
        """Load an optional installation-level artifact configuration."""
        manifest = os.environ.get("GRADFLOW_DVEB_ARTIFACT")
        if not manifest:
            return None
        return cls.from_manifest(
            manifest,
            model=os.environ.get("GRADFLOW_DVEB_MODEL"),
            verified_model_sha256=os.environ.get("GRADFLOW_DVEB_MODEL_SHA256"),
        )

    def _validate_model_record(self) -> None:
        if (self.model is None) != (self.verified_model_sha256 is None):
            raise ValueError(
                "DVEB model path and verified hash must be supplied together"
            )
        if self.model is None:
            return
        lines = self.model.read_text().splitlines(keepends=True)
        if not lines or not lines[-1].startswith("model_sha256\t"):
            raise ValueError("DVEB placement model has no integrity hash")
        recorded = lines[-1].strip().split("\t", 1)[1]
        actual = hashlib.sha256("".join(lines[:-1]).encode()).hexdigest()
        if recorded != actual or actual != self.verified_model_sha256:
            raise ValueError("DVEB placement model integrity hash differs")
        fields = {}
        for line in lines[:-1]:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                fields[parts[0]] = parts[1]
        if fields.get("schema") != "dveb-placement-v1":
            raise ValueError("unsupported DVEB placement model schema")
        if fields.get("program_sha256") != self.program_sha256:
            raise ValueError("DVEB placement model belongs to another program")
        if fields.get("module_sha256") != self.module_sha256:
            raise ValueError("DVEB placement model belongs to another math module")


@dataclass(frozen=True)
class DvebRunResult:
    state: Tensor
    selected_target: str
    selected_cpu_workers: int
    execution_seconds: float
    total_seconds: float
    peak_bytes: int


@dataclass(frozen=True)
class DvebDeviceRunResult:
    """One completed synchronous ABI v2 invocation."""

    state: Tensor
    execution_seconds: float
    total_seconds: float
    workspace_bytes: int


class DvebPortableAbi:
    """Loaded ABI v1 artifact. Construction and calls verify identities."""

    def __init__(self, artifact: DvebArtifact) -> None:
        if _sha256(artifact.library) != artifact.library_sha256:
            raise ValueError("DVEB ABI library changed after configuration")
        self.artifact = artifact
        self._library = ctypes.CDLL(str(artifact.library))
        self._library.dveb_portable_abi_version.restype = ctypes.c_uint32
        self._library.dveb_portable_query_v1.argtypes = [
            ctypes.c_int32, ctypes.POINTER(_Query), ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.dveb_portable_query_v1.restype = ctypes.c_int
        self._library.dveb_portable_run_v1.argtypes = [
            ctypes.POINTER(_Request), ctypes.POINTER(_Result), ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.dveb_portable_run_v1.restype = ctypes.c_int
        if self._library.dveb_portable_abi_version() != ABI_VERSION:
            raise ValueError("DVEB shared library does not implement ABI v1")

    def query(self, intervals: int) -> dict[str, int | str]:
        query = _Query(struct_size=ctypes.sizeof(_Query), abi_version=ABI_VERSION)
        error = ctypes.create_string_buffer(512)
        status = self._library.dveb_portable_query_v1(
            intervals, ctypes.byref(query), error, len(error)
        )
        if status != _OK:
            raise DvebAbiError(status, error.value.decode())
        program = query.program_sha256.decode()
        module = query.module_sha256.decode()
        if program != self.artifact.program_sha256:
            raise ValueError("loaded DVEB library has the wrong program identity")
        if module != self.artifact.module_sha256:
            raise ValueError("loaded DVEB library has the wrong module identity")
        return {
            "scalar_type": query.scalar_type,
            "dimensions": query.dimensions,
            "components": query.components,
            "required_elements": query.required_elements,
            "program_sha256": program,
            "module_sha256": module,
        }

    def run(
        self,
        state: Tensor,
        *,
        steps: int,
        target: str = "auto",
        cpu_workers: int = 1,
    ) -> DvebRunResult:
        if state.device.type != "cpu" or state.dtype is not torch.float32:
            raise ValueError("DVEB ABI v1 requires a caller-owned CPU float32 tensor")
        if state.requires_grad:
            raise ValueError("DVEB ABI v1 is not an autograd backend")
        if not state.is_contiguous():
            raise ValueError("DVEB ABI v1 requires contiguous component-major state")
        if state.ndim != 4 or state.shape[0] != 5:
            raise ValueError("DVEB ABI v1 state layout is (5, nz+1, ny+1, nx+1)")
        if len(set(state.shape[1:])) != 1:
            raise ValueError("this compiled DVEB pipeline requires a cubic grid")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("DVEB ABI v1 requires a positive fixed step count")
        target_key = target.lower().replace("_", "-")
        target_key = {"cpu-simd": "cpu", "cuda-native": "cuda"}.get(
            target_key, target_key
        )
        if target_key not in _TARGETS:
            raise ValueError(f"unknown DVEB target {target!r}")
        if target_key == "cpu" and (
            isinstance(cpu_workers, bool) or not isinstance(cpu_workers, int)
            or not 1 <= cpu_workers <= 256
        ):
            raise ValueError("cpu_workers must be in 1..256")
        if target_key == "auto" and self.artifact.model is None:
            raise ValueError("automatic DVEB placement requires a verified model")
        if target_key == "auto":
            # Do not turn a model changed after configuration into a benign
            # bounded-placement fallback.
            self.artifact._validate_model_record()

        intervals = state.shape[-1] - 1
        metadata = self.query(intervals)
        if metadata["required_elements"] != state.numel():
            raise ValueError("DVEB ABI query disagrees with the caller state shape")
        output = torch.empty_like(state)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        model_bytes = (
            str(self.artifact.model).encode() if self.artifact.model else None
        )
        model_hash_bytes = (
            self.artifact.verified_model_sha256.encode()
            if self.artifact.verified_model_sha256 else None
        )
        request = _Request(
            struct_size=ctypes.sizeof(_Request), abi_version=ABI_VERSION,
            intervals=intervals, steps=steps, target=_TARGETS[target_key],
            cpu_workers=cpu_workers, endpoint=2,
            input=ctypes.cast(state.data_ptr(), float_pointer),
            input_count=state.numel(),
            output=ctypes.cast(output.data_ptr(), float_pointer),
            output_capacity=output.numel(), model_path=model_bytes,
            verified_model_sha256=model_hash_bytes,
        )
        result = _Result(
            struct_size=ctypes.sizeof(_Result), abi_version=ABI_VERSION
        )
        error = ctypes.create_string_buffer(512)
        status = self._library.dveb_portable_run_v1(
            ctypes.byref(request), ctypes.byref(result), error, len(error)
        )
        if status != _OK:
            raise DvebAbiError(status, error.value.decode())
        if result.output_count != output.numel() or result.finite != 1:
            raise RuntimeError("DVEB ABI returned invalid result metadata")
        return DvebRunResult(
            state=output,
            selected_target=_TARGET_NAMES[result.selected_target],
            selected_cpu_workers=result.selected_cpu_workers,
            execution_seconds=result.execution_seconds,
            total_seconds=result.total_seconds,
            peak_bytes=result.peak_bytes,
        )


class DvebDeviceContext:
    """Reusable device-resident ABI v2 context for one cubic grid and GPU.

    Calls use the current PyTorch CUDA stream and return only after that stream
    has completed. The context owns DVEB workspace but never owns caller state.
    A context is deliberately not safe for concurrent calls.
    """

    def __init__(
        self,
        artifact: DvebArtifact,
        intervals: int,
        *,
        device: torch.device | str | int | None = None,
    ) -> None:
        if artifact.device_library is None or artifact.device_library_sha256 is None:
            raise ValueError("DVEB artifact does not provide device ABI v2")
        if _sha256(artifact.device_library) != artifact.device_library_sha256:
            raise ValueError("DVEB device ABI library changed after configuration")
        if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals < 4:
            raise ValueError("intervals must be an integer of at least four")
        if not torch.cuda.is_available():
            raise RuntimeError("DVEB device ABI v2 requires CUDA")
        if device is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            resolved = torch.device("cuda", device)
        else:
            resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("DVEB device ABI v2 requires a CUDA device")
        ordinal = torch.cuda.current_device() if resolved.index is None else resolved.index
        assert ordinal is not None

        self.artifact = artifact
        self.intervals = intervals
        self.device = torch.device("cuda", ordinal)
        self._library = ctypes.CDLL(str(artifact.device_library))
        self._library.dveb_portable_device_abi_version.restype = ctypes.c_uint32
        self._library.dveb_portable_device_create_v2.argtypes = [
            ctypes.POINTER(_DeviceCreateRequest), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_DeviceCreateResult), ctypes.c_char_p, ctypes.c_size_t,
        ]
        self._library.dveb_portable_device_create_v2.restype = ctypes.c_int
        self._library.dveb_portable_device_run_v2.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_DeviceRunRequest),
            ctypes.POINTER(_DeviceRunResult), ctypes.c_char_p, ctypes.c_size_t,
        ]
        self._library.dveb_portable_device_run_v2.restype = ctypes.c_int
        self._library.dveb_portable_device_destroy_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ]
        self._library.dveb_portable_device_destroy_v2.restype = ctypes.c_int
        if self._library.dveb_portable_device_abi_version() != DEVICE_ABI_VERSION:
            raise ValueError("DVEB shared library does not implement device ABI v2")

        request = _DeviceCreateRequest(
            struct_size=ctypes.sizeof(_DeviceCreateRequest),
            abi_version=DEVICE_ABI_VERSION,
            intervals=intervals,
            device_ordinal=ordinal,
        )
        result = _DeviceCreateResult(
            struct_size=ctypes.sizeof(_DeviceCreateResult),
            abi_version=DEVICE_ABI_VERSION,
        )
        context = ctypes.c_void_p()
        error = ctypes.create_string_buffer(512)
        with torch.cuda.device(self.device):
            status = self._library.dveb_portable_device_create_v2(
                ctypes.byref(request), ctypes.byref(context), ctypes.byref(result),
                error, len(error),
            )
        if status != _OK:
            raise DvebAbiError(status, error.value.decode())
        if not context.value or result.device_ordinal != ordinal:
            raise RuntimeError("DVEB device ABI returned invalid context metadata")
        self._context: ctypes.c_void_p | None = context
        self.required_elements = int(result.required_elements)
        self.workspace_bytes = int(result.workspace_bytes)

    def _validate_state(self, state: Tensor, label: str) -> None:
        if state.device != self.device or state.dtype is not torch.float32:
            raise ValueError(f"{label} must be CUDA float32 on {self.device}")
        if state.requires_grad:
            raise ValueError("DVEB device ABI v2 is not an autograd backend")
        if not state.is_contiguous():
            raise ValueError(f"{label} must be contiguous component-major state")
        side = self.intervals + 1
        if tuple(state.shape) != (5, side, side, side):
            raise ValueError(f"{label} layout must be (5, {side}, {side}, {side})")

    def run(
        self,
        state: Tensor,
        *,
        steps: int,
        out: Tensor | None = None,
    ) -> DvebDeviceRunResult:
        if self._context is None:
            raise RuntimeError("DVEB device context is closed")
        self._validate_state(state, "state")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("DVEB device ABI v2 requires a positive fixed step count")
        output = torch.empty_like(state) if out is None else out
        self._validate_state(output, "out")
        float_pointer = ctypes.POINTER(ctypes.c_float)
        stream = torch.cuda.current_stream(self.device)
        request = _DeviceRunRequest(
            struct_size=ctypes.sizeof(_DeviceRunRequest),
            abi_version=DEVICE_ABI_VERSION,
            steps=steps,
            input_device=ctypes.cast(state.data_ptr(), float_pointer),
            input_count=state.numel(),
            output_device=ctypes.cast(output.data_ptr(), float_pointer),
            output_capacity=output.numel(),
            cuda_stream=ctypes.c_void_p(stream.cuda_stream),
        )
        result = _DeviceRunResult(
            struct_size=ctypes.sizeof(_DeviceRunResult),
            abi_version=DEVICE_ABI_VERSION,
        )
        error = ctypes.create_string_buffer(512)
        with torch.cuda.device(self.device):
            status = self._library.dveb_portable_device_run_v2(
                self._context, ctypes.byref(request), ctypes.byref(result),
                error, len(error),
            )
        if status != _OK:
            raise DvebAbiError(status, error.value.decode())
        if (result.synchronized != 1 or result.output_count != output.numel()
                or result.device_ordinal != self.device.index):
            raise RuntimeError("DVEB device ABI returned invalid run metadata")
        return DvebDeviceRunResult(
            state=output,
            execution_seconds=result.execution_seconds,
            total_seconds=result.total_seconds,
            workspace_bytes=result.workspace_bytes,
        )

    def close(self) -> None:
        context, self._context = self._context, None
        if context is None:
            return
        error = ctypes.create_string_buffer(512)
        with torch.cuda.device(self.device):
            status = self._library.dveb_portable_device_destroy_v2(
                context, error, len(error)
            )
        if status != _OK:
            raise DvebAbiError(status, error.value.decode())

    def __enter__(self) -> "DvebDeviceContext":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
