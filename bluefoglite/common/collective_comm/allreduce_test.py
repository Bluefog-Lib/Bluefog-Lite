import sys

import numpy as np  # type: ignore
import pytest  # type: ignore

from bluefoglite.common.collective_comm.allreduce import allreduce_tree
from bluefoglite.common.tcp.buffer import NumpyBuffer
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.testing.util import multi_process_help
from bluefoglite.testing.fixture import fixture_store


@pytest.fixture(name="store")
def fixture_store_wrapper():
    yield from fixture_store(__name__)


# See https://github.com/spack/spack/issues/14102 as example
@pytest.mark.skipif(
    sys.platform == "darwin" and sys.version_info[:2] == (3, 8),
    reason="Can't pickle local object in multiprocess",
)
@pytest.mark.parametrize("size", [2, 3, 4, 5, 6, 9, 11, 16])
def test_allreduce_tree(store, size):
    dim = 10

    def allreduce(rank, size):
        agent = Agent()
        array = np.array(range(dim), dtype=np.float64) + rank
        context = agent.createContext(rank=rank, size=size)
        context.connectFull(store)
        buf = NumpyBuffer(context, array=array)
        allreduce_tree(buf=buf, context=context)
        expected_array = np.array(range(dim), dtype=np.float64) + (size - 1) / 2
        np.testing.assert_allclose(array, expected_array)

    errors = multi_process_help(size=size, fn=allreduce)
    store.reset()
    for error in errors:
        raise error
