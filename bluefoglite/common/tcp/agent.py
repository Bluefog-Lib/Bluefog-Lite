# Copyright 2021 Bluefog Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import copy
import math
import socket
from typing import Callable, Dict, Optional, List, Tuple

from bluefoglite.common.tcp.eventloop import EventLoop
from bluefoglite.common.tcp.pair import Pair, SocketFullAddress, TAddress
from bluefoglite.common.logger import Logger

# One agent can contain multiple Contexts.
# Each Context should represent entire communication group like  (comm in MPI)
# In each Context, it contains multiple Pairs, i.e. socket pair, talking to other neighbor.
# Op `send` and `recv`` should be attached to a fixed memory buffer.
class Agent:
    def __init__(self, address: Optional[SocketFullAddress] = None):
        # One event loop process the events of all socket pairs
        self.event_loop: EventLoop = EventLoop()
        self.context: Optional[AgentContext] = None

        self.event_loop.run()

    def __del__(self):
        self.close()

    def createAgentAddress(  # pylint: disable=no-self-use
        self, *, addr: Optional[TAddress] = None
    ) -> SocketFullAddress:
        if addr is None:
            addr = ("localhost", 0)  # let the OS to pick a random free address
        return SocketFullAddress(
            addr=addr,
            sock_family=socket.AF_INET,
            sock_type=socket.SOCK_STREAM,
            sock_protocol=socket.IPPROTO_IP,
        )

    def createContext(
        self, *, rank: int, size: int, addr: Optional[TAddress] = None
    ) -> "AgentContext":
        full_address = self.createAgentAddress(addr=addr)
        self.context = AgentContext(
            event_loop=self.event_loop, rank=rank, size=size, full_address=full_address
        )
        return self.context

    def close(self):
        if self.context:
            self.context.close()
        if self.event_loop.is_alive():
            self.event_loop.close()


class AgentContext:
    def __init__(
        self,
        *,
        event_loop: EventLoop,
        rank: int,
        size: int,
        full_address: SocketFullAddress,
    ):
        self.rank = rank
        self.size = size
        self.full_address = full_address
        self.pairs: Dict[int, "Pair"] = {}
        self._event_loop = event_loop

    def getPair(self, peer_rank) -> Pair:
        if peer_rank not in self.pairs:
            raise KeyError(f"Cannot find the Pair for {(self.rank, peer_rank)}")
        return self.pairs[peer_rank]

    def getOrCreatePair(self, peer_rank) -> Pair:
        if peer_rank not in self.pairs:
            self.pairs[peer_rank] = self.createPair(peer_rank)
        return self.pairs[peer_rank]

    def createPair(self, peer_rank) -> Pair:
        pair = Pair(
            event_loop=self._event_loop,
            self_rank=self.rank,
            peer_rank=peer_rank,
            full_address=copy.copy(self.full_address),  # important!
        )
        self.pairs[peer_rank] = pair
        return pair

    def close(self) -> None:
        for _, pair in self.pairs.items():
            pair.close()
        self.pairs.clear()

    def connectFull(self, store) -> None:
        def full_neighbor_fn(self_rank, peer_rank, size):
            del self_rank, peer_rank, size
            return True

        self._connectGivenNeighborFunc(store, full_neighbor_fn)

    def connectRing(self, store) -> None:
        def ring_neighbor_fn(self_rank, peer_rank, size):
            if peer_rank == (self_rank + 1) % size:
                return True
            if peer_rank == (self_rank - 1) % size:
                return True
            return False

        self._connectGivenNeighborFunc(store, ring_neighbor_fn)

    def connectHypercube(self, store) -> None:
        def hypercube_neighbor_fn(self_rank, peer_rank, size):
            dim = math.ceil(math.log2(size))
            for i in range(dim):
                if self_rank ^ (1 << i) == peer_rank:
                    return True
            return False

        self._connectGivenNeighborFunc(store, hypercube_neighbor_fn)

    def connectExponentialTwo(self, store) -> None:
        def expo2_neighbor_fn(self_rank, peer_rank, size):
            diff = abs(self_rank - peer_rank)
            diff2 = size - abs(self_rank - peer_rank)
            if diff & (diff - 1) == 0:
                return True
            if diff2 & (diff2 - 1) == 0:
                return True
            return False

        self._connectGivenNeighborFunc(store, expo2_neighbor_fn)

    def _connectGivenNeighborFunc(
        self, store, neighbor_fn: Callable[[int, int, int], bool]
    ) -> None:
        # It store the self address listening for the other ranks.
        _all_address: List[Optional[SocketFullAddress]] = []

        for i in range(self.size):
            is_neighbor = neighbor_fn(self.rank, i, self.size)
            if i == self.rank or not is_neighbor:
                # Just placeholder
                _all_address.append(None)
                continue

            if not neighbor_fn(i, self.rank, self.size):
                raise ValueError(
                    "The connection function must be symmetry, "
                    "i.e., neighbor_fn(i, j, size) ==  neighbor_fn(j, i, size)"
                )
            pair = self.createPair(i)
            _all_address.append(pair.self_address)

        Logger.get().debug("start listening done")

        # we need a global store to share between all processes.
        store.set(f"rank_addr_{self.rank}", _all_address)

        # Connect others
        for i in range(self.size):
            is_neighbor = neighbor_fn(self.rank, i, self.size)
            if i == self.rank or not is_neighbor:
                continue
            # Other's file store
            others_all_address = store.get(f"rank_addr_{i}")
            # The listening address open for self.
            addr = others_all_address[self.rank]
            pair = self.getPair(i)
            pair.connect(addr=addr)

        Logger.get().debug("connect pairs done")
