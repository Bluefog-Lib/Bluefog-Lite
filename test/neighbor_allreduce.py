import bluefoglite as bfl
import numpy as np


def setup_module(module):
    bfl.init()


def teardown_module(module):
    # Why do we need to call shutdown explicitly?
    # We need to detect event_loop thread is_alive or not.
    bfl.shutdown()


def test_neighbor_allreduce_exp2():
    rank, size = bfl.rank(), bfl.size()
    assert size > 3

    num_indegree = int(np.ceil(np.log2(size)))
    uniform_weight = 1 / (num_indegree + 1)
    src_neighbor_ranks = [(rank - 2 ** i) % size for i in range(num_indegree)]
    dst_neighbor_ranks = [(rank + 2 ** i) % size for i in range(num_indegree)]

    sum_value = np.sum(src_neighbor_ranks) + rank
    dst_weights = {r: 1 for r in dst_neighbor_ranks}
    src_weights = {r: uniform_weight for r in src_neighbor_ranks}

    array = np.array([1.0, 2, 3, 4]) + bfl.rank()
    ret_array = bfl.neighbor_allreduce(
        array,
        self_weight=uniform_weight,
        dst_weights=dst_weights,
        src_weights=src_weights,
    )
    expected_array = np.array([1.0, 2, 3, 4]) + sum_value * uniform_weight
    np.testing.assert_allclose(ret_array, expected_array)
