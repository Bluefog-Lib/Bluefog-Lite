import datetime
import functools
import os
import sys

import numpy as np  # type: ignore
import pytest  # type: ignore

from bluefoglite.common.collective_comm.broadcast import (
    broadcast_one_to_all,
    broadcast_spreading,
)
from bluefoglite.common.store import FileStore
from bluefoglite.common.tcp.buffer import SpecifiedBuffer
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.testing.util import multi_process_help


@pytest.fixture(name="store")
def fixture_store():
    runtime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    shared_file_dir = os.path.join("/tmp", ".bluefoglite", __name__, runtime_str)
    if not os.path.exists(shared_file_dir):
        os.makedirs(shared_file_dir)
    f_store = FileStore(shared_file_dir)
    yield f_store
    f_store.close()


# See https://github.com/spack/spack/issues/14102 as example
@pytest.mark.skipif(
    sys.platform == "darwin" and sys.version_info[:2] == (3, 8),
    reason="Can't pickle local object in multiprocess",
)
@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_broadcast_one_to_all(store, size):
    def broadcast(rank, size, dim, root_rank):
        agent = Agent()
        array = np.array([rank] * dim)
        context = agent.createContext(rank=rank, size=size)
        context.connectFull(store)
        buf = SpecifiedBuffer(
            context, buffer_view=array.data, buffer_length=array.nbytes
        )
        broadcast_one_to_all(buf=buf, root_rank=root_rank, context=context)
        np.testing.assert_allclose(array, np.array([root_rank] * dim))

    dim = 10
    root_rank = 0
    _broadcast = functools.partial(broadcast, dim=dim, root_rank=root_rank)

    errors = multi_process_help(size=size, fn=_broadcast)
    for error in errors:
        raise error


# See https://github.com/spack/spack/issues/14102 as example
@pytest.mark.skipif(
    sys.platform == "darwin" and sys.version_info[:2] == (3, 8),
    reason="Can't pickle local object in multiprocess",
)
@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_broadcast_spreading(store, size):
    def broadcast(rank, size, dim, root_rank):
        agent = Agent()
        array = np.array([rank] * dim)
        context = agent.createContext(rank=rank, size=size)
        context.connectFull(store)
        buf = SpecifiedBuffer(
            context, buffer_view=array.data, buffer_length=array.nbytes
        )
        broadcast_spreading(buf=buf, root_rank=root_rank, context=context)
        np.testing.assert_allclose(array, np.array([root_rank] * dim))

    dim = 10
    root_rank = 0
    _broadcast = functools.partial(broadcast, dim=dim, root_rank=root_rank)

    errors = multi_process_help(size=size, fn=_broadcast)
    for error in errors:
        raise error
