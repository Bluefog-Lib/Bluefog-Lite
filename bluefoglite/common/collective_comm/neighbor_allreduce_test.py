import itertools
import sys

import numpy as np
import pytest

from bluefoglite.common.collective_comm.neighbor_allreduce import neighbor_allreduce
from bluefoglite.common.tcp.buffer import NumpyBuffer
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.testing.fixture import fixture_store
from bluefoglite.testing.util import multi_process_help


@pytest.fixture(name="store")
def fixture_store_wrapper():
    yield from fixture_store(__name__)


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 8) and sys.platform == "darwin",
    reason="Python 3.8 in Mac can't pickle local object in multiprocess",
)
@pytest.mark.parametrize(
    "size,dtype,ndim",
    itertools.product([2, 3, 4, 6], [np.float32, np.float64, np.float128], [1, 2, 3]),
)
def test_neighbor_allreduce_ring(store, size, dtype, ndim):
    dim = 7

    def nar(rank, size):
        agent = Agent()
        array = np.array(range(dim**ndim), dtype=dtype).reshape([dim] * ndim) + rank
        context = agent.createContext(rank=rank, size=size)
        context.connectFull(store)
        in_buf = NumpyBuffer(context, array=array)
        out_buf = in_buf.clone()
        l_r = (rank + 1) % size
        r_r = (rank - 1) % size
        dst_weights = {l_r: 1}
        src_weights = {r_r: 1 / 2}
        neighbor_allreduce(
            in_buf=in_buf,
            out_buf=out_buf,
            self_weight=1 / 2,
            src_weights=src_weights,
            dst_weights=dst_weights,
        )
        expected_array = (
            np.array(range(dim**ndim), dtype=dtype).reshape([dim] * ndim)
            + (r_r + rank) / 2
        )
        np.testing.assert_allclose(out_buf.array, expected_array)

    errors = multi_process_help(size=size, fn=nar, timeout=10)
    store.reset()
    for error in errors:
        raise error


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 8) and sys.platform == "darwin",
    reason="Python 3.8 in Mac can't pickle local object in multiprocess",
)
@pytest.mark.parametrize(
    "size,dtype,ndim",
    itertools.product([3, 4, 6], [np.float32, np.float64, np.float128], [1, 2, 3]),
)
def test_neighbor_allreduce_exp(store, size, dtype, ndim):
    dim = 7

    def nar(rank, size):
        agent = Agent()
        array = np.array(range(dim**ndim), dtype=dtype).reshape([dim] * ndim) + rank
        context = agent.createContext(rank=rank, size=size)
        context.connectFull(store)
        in_buf = NumpyBuffer(context, array=array)
        out_buf = in_buf.clone()

        # Exponential 2 topology
        num_indegree = int(np.ceil(np.log2(size)))
        uniform_weight = 1 / (num_indegree + 1)
        src_neighbor_ranks = [(rank - 2**i) % size for i in range(num_indegree)]
        dst_neighbor_ranks = [(rank + 2**i) % size for i in range(num_indegree)]
        sum_value = np.sum(src_neighbor_ranks) + rank
        dst_weights = {r: 1 for r in dst_neighbor_ranks}
        src_weights = {r: uniform_weight for r in src_neighbor_ranks}

        neighbor_allreduce(
            in_buf=in_buf,
            out_buf=out_buf,
            self_weight=uniform_weight,
            src_weights=src_weights,
            dst_weights=dst_weights,
        )
        expected_array = (
            np.array(range(dim**ndim), dtype=dtype).reshape([dim] * ndim)
            + sum_value * uniform_weight
        )
        np.testing.assert_allclose(
            out_buf.array, expected_array, rtol=1e-4 if dtype == np.float32 else 1e-7
        )

    errors = multi_process_help(size=size, fn=nar, timeout=10)
    store.reset()
    for error in errors:
        raise error
