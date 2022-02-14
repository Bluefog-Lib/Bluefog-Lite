import math
from unittest.mock import MagicMock, patch
import pytest  # type: ignore

from bluefoglite.common.store import InMemoryStore
from bluefoglite.common.tcp.agent import AgentContext

from bluefoglite.testing.util import multi_thread_help


def mocked_create_pair(self, peer_rank):
    pair = MagicMock()
    pair.self_address = f"{self.rank}<-{peer_rank}"
    pair.connect = MagicMock()
    self.pairs[peer_rank] = pair
    return pair


@pytest.mark.parametrize("size", [1, 2, 3, 5, 9])
def test_connect_ring(size):
    store = InMemoryStore()

    def fn(rank, size):
        with patch.object(AgentContext, "createPair", new=mocked_create_pair):
            context = AgentContext(
                event_loop=MagicMock(), rank=rank, size=size, full_address=MagicMock()
            )
            # Patch createPair, getPair
            context.connectRing(store)
            assert len(context.pairs) == min(2, size - 1)
            # Check if the addr is the right one
            for peer, pair in context.pairs.items():
                if peer == rank:
                    assert pair is None
                # Each pair should call connect only once.
                assert len(pair.connect.call_args_list) == 1
                pair.connect.assert_called_with(addr=f"{peer}<-{rank}")

    errors = multi_thread_help(size=size, fn=fn)

    # TODO check the value stored in the store.
    # print("Value in store: ", store.store)

    for e in errors:
        raise e


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8])
def test_connect_full(size):
    store = InMemoryStore()

    def fn(rank, size):
        with patch.object(AgentContext, "createPair", new=mocked_create_pair):
            context = AgentContext(
                event_loop=MagicMock(), rank=rank, size=size, full_address=MagicMock()
            )
            context.connectFull(store)
            assert len(context.pairs) == size - 1
            # Check if the addr is the right one
            for peer, pair in context.pairs.items():
                if peer == rank:
                    assert pair is None
                # Each pair should call connect only once.
                assert len(pair.connect.call_args_list) == 1
                pair.connect.assert_called_with(addr=f"{peer}<-{rank}")

    errors = multi_thread_help(size=size, fn=fn)

    # TODO check the value stored in the store.
    # print("Value in store: ", store.store)

    for e in errors:
        raise e


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])  # what if it is not the power of 2?
def test_connect_hypercube(size):
    store = InMemoryStore()

    def fn(rank, size):
        with patch.object(AgentContext, "createPair", new=mocked_create_pair):
            context = AgentContext(
                event_loop=MagicMock(), rank=rank, size=size, full_address=MagicMock()
            )
            # Patch createPair, getPair
            context.connectHypercube(store)
            assert len(context.pairs) == math.floor(math.log2(size))
            # Check if the addr is the right one
            for peer, pair in context.pairs.items():
                if peer == rank:
                    assert pair is None
                # Each pair should call connect only once.
                assert len(pair.connect.call_args_list) == 1
                pair.connect.assert_called_with(addr=f"{peer}<-{rank}")

    errors = multi_thread_help(size=size, fn=fn)

    # TODO check the value stored in the store.
    # print("Value in store: ", store.store)

    for e in errors:
        raise e


@pytest.mark.parametrize(
    "size,expected_pairs",
    [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 4),
        (7, 6),
        (8, 5),
        # (16, 7),
        # (27, 10),
        # (32, 9),
    ],
)
def test_connect_exponential2(size, expected_pairs):
    store = InMemoryStore()

    def fn(rank, size):
        with patch.object(AgentContext, "createPair", new=mocked_create_pair):
            context = AgentContext(
                event_loop=MagicMock(), rank=rank, size=size, full_address=MagicMock()
            )
            # Patch createPair, getPair
            context.connectExponentialTwo(store)
            assert len(context.pairs) == expected_pairs
            # Check if the addr is the right one
            for peer, pair in context.pairs.items():
                if peer == rank:
                    assert pair is None
                # Each pair should call connect only once.
                assert len(pair.connect.call_args_list) == 1
                pair.connect.assert_called_with(addr=f"{peer}<-{rank}")

    errors = multi_thread_help(size=size, fn=fn, timeout=15)

    # TODO check the value stored in the store.
    # print("Value in store: ", store.store)

    for e in errors:
        raise e
