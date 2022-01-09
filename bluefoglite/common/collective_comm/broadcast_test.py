import numpy as np  # type: ignore

from bluefoglite.common.collective_comm.broadcast import (
    broadcast_one_to_all,
    broadcast_spreading,
)
from bluefoglite.common.store import InMemoryStore
from bluefoglite.common.tcp.buffer import SpecifiedBuffer
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.testing.util import multi_thread_help


def test_broadcast_one_to_all():
    dim = 10
    root_rank = 0
    size = 2
    store = InMemoryStore()

    def broadcast(rank, size):
        agent = Agent()
        array = np.array([rank] * dim)
        context = agent.createContext(rank=rank, size=size)
        context.connectRing(store)
        buf = SpecifiedBuffer(
            context, buffer_view=array.data, buffer_length=array.nbytes
        )
        broadcast_one_to_all(buf=buf, root_rank=root_rank, context=context)
        np.testing.assert_allclose(array, np.array([root_rank] * dim))

    errors = multi_thread_help(size=size, fn=broadcast)

    for error in errors:
        raise error


def test_broadcast_spreading():
    dim = 10
    root_rank = 0
    size = 2
    store = InMemoryStore()

    def broadcast(rank, size):
        agent = Agent()
        array = np.array([rank] * dim)
        context = agent.createContext(rank=rank, size=size)
        context.connectRing(store)
        buf = SpecifiedBuffer(
            context, buffer_view=array.data, buffer_length=array.nbytes
        )
        broadcast_spreading(buf=buf, root_rank=root_rank, context=context)
        np.testing.assert_allclose(array, np.array([root_rank] * dim))

    errors = multi_thread_help(size=size, fn=broadcast)

    for error in errors:
        raise error
