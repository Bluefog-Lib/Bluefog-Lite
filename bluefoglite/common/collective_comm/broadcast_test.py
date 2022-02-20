import functools
import sys

import numpy as np
import pytest

from bluefoglite.common.collective_comm.broadcast import (
    broadcast_one_to_all,
    broadcast_ring,
    broadcast_spreading,
)
from bluefoglite.common.tcp.buffer import SpecifiedBuffer
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.testing.fixture import fixture_store
from bluefoglite.testing.util import multi_process_help


@pytest.fixture(name="store", scope="function")
def fixture_store_wrapper():
    yield from fixture_store(__name__)


# See https://github.com/spack/spack/issues/14102 as example
@pytest.mark.skipif(
    sys.platform == "darwin" and sys.version_info[:2] == (3, 8),
    reason="Can't pickle local object in multiprocess",
)
@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_broadcast_one_to_all(store, size):
    def broadcast(rank, size, dim, root_rank):
        agent = Agent()
        array = np.array(range(dim)) + rank
        context = agent.createContext(rank=rank, size=size)
        context.connectFull(store)
        buf = SpecifiedBuffer(
            context, buffer_view=array.data.cast("c"), buffer_length=array.nbytes
        )
        broadcast_one_to_all(buf=buf, root_rank=root_rank, context=context)
        np.testing.assert_allclose(array, np.array(range(dim)) + root_rank)

    dim = 10
    root_rank = 0
    _broadcast = functools.partial(broadcast, dim=dim, root_rank=root_rank)

    errors = multi_process_help(size=size, fn=_broadcast)
    store.reset()
    for error in errors:
        raise error


@pytest.mark.skipif(
    sys.platform == "darwin" and sys.version_info[:2] == (3, 8),
    reason="Can't pickle local object in multiprocess",
)
@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_broadcast_ring(store, size):
    def broadcast(rank, size, dim, root_rank):
        agent = Agent()
        array = np.array(range(dim)) + rank
        context = agent.createContext(rank=rank, size=size)
        context.connectRing(store)
        buf = SpecifiedBuffer(
            context, buffer_view=array.data.cast("c"), buffer_length=array.nbytes
        )
        broadcast_ring(buf=buf, root_rank=root_rank, context=context)
        np.testing.assert_allclose(array, np.array(range(dim)) + root_rank)

    dim = 10
    root_rank = 0
    _broadcast = functools.partial(broadcast, dim=dim, root_rank=root_rank)

    errors = multi_process_help(size=size, fn=_broadcast)
    store.reset()
    for error in errors:
        raise error


@pytest.mark.skip(
    "Encountered when recv: [Errno 32] Broken pipe."
    "Likely, the other side of socket closed connection."
)
# See https://github.com/spack/spack/issues/14102 as example
@pytest.mark.skipif(
    sys.platform == "darwin" and sys.version_info[:2] == (3, 8),
    reason="Can't pickle local object in multiprocess",
)
@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_broadcast_spreading(store, size):
    def broadcast(rank, size, dim, root_rank):
        agent = Agent()
        array = np.array(range(dim)) + rank
        context = agent.createContext(rank=rank, size=size)
        context.connectFull(store)
        buf = SpecifiedBuffer(
            context, buffer_view=array.data.cast("c"), buffer_length=array.nbytes
        )
        broadcast_spreading(buf=buf, root_rank=root_rank, context=context)
        np.testing.assert_allclose(array, np.array(range(dim)) + root_rank)

    dim = 10
    root_rank = 0
    _broadcast = functools.partial(broadcast, dim=dim, root_rank=root_rank)

    errors = multi_process_help(size=size, fn=_broadcast)
    for error in errors:
        raise error
