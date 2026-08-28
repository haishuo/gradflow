from experiments.deferred_cuda_gates.verify import main as verify_deferred_cuda


def test_deferred_cuda_correctness_record_verifies() -> None:
    verify_deferred_cuda()
