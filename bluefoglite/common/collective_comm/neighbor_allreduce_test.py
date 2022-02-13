import itertools
import numpy as np  # type: ignore
import pytest  # type: ignore

from bluefoglite.common.collective_comm.neighbor_allreduce import neighbor_allreduce
from bluefoglite.common.tcp.buffer import NumpyBuffer
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.testing.fixture import fixture_store
from bluefoglite.testing.util import multi_process_help


@pytest.fixture(name="store")
def fixture_store_wrapper():
    yield from fixture_store(__name__)


@pytest.mark.parametrize(
    "size,dtype,ndim",
    itertools.product([2, 3, 4, 6], [np.float32, np.float64, np.float128], [1, 2, 3]),
)
def test_neighbor_allreduce_ring(store, size, dtype, ndim):
    dim = 7

    def nar(rank, size):
        agent = Agent()
        array = np.array(range(dim ** ndim), dtype=dtype).reshape([dim] * ndim) + rank
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
            np.array(range(dim ** ndim), dtype=dtype).reshape([dim] * ndim)
            + (r_r + rank) / 2
        )
        np.testing.assert_allclose(out_buf.array, expected_array)

    errors = multi_process_help(size=size, fn=nar, timeout=10000)
    store.reset()
    for error in errors:
        raise error
