from experiments.fd_fv_bakeoff.verify_phase_4a import main as verify_phase_4a
from experiments.fd_fv_bakeoff.verify_phase_4b import main as verify_phase_4b
from experiments.fd_fv_bakeoff.verify_phase_4r import main as verify_phase_4r
from experiments.fd_fv_bakeoff.verify_phase_4r_cuda import (
    main as verify_phase_4r_cuda,
)


def test_phase_4a_admission_record_verifies() -> None:
    verify_phase_4a()


def test_phase_4b_benchmark_record_verifies() -> None:
    verify_phase_4b()


def test_phase_4r_replication_record_verifies() -> None:
    verify_phase_4r()


def test_phase_4r_cuda_replication_record_verifies() -> None:
    verify_phase_4r_cuda()
