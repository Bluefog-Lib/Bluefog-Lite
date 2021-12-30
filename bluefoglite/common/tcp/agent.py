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


import dataclasses
import socket
from typing import Dict, Optional

from bluefoglite.common.tcp.eventloop import EventLoop
from bluefoglite.common.tcp.pair import Pair, SocketAddress
from bluefoglite.common.logger import logger

BASE_PART = 18106  # Looks like 1BL[UEUF]OG

# One agent can contain multiple Contexts.
# Each Context should represent entire communication group like  (comm in MPI)
# In each Context, it contains multiple Pairs, i.e. socket pair, talking to other neighbor.
# Op `send` and `recv`` should be attached to a fixed memory buffer.
class Agent(object):
    def __init__(self, address: Optional[SocketAddress] = None):
        # One event loop process the events of all socket pairs
        self.event_loop: EventLoop = EventLoop()
        self.context = None

        self.event_loop.run()

    def createAgentAddress(self, *, rank, size) -> SocketAddress:
        return SocketAddress(
            addr=("localhost", BASE_PART + rank * size),
            sock_family=socket.AF_INET,
            sock_type=socket.SOCK_STREAM,
            sock_protocol=0,
        )

    def createContext(self, *, rank, size):
        address = self.createAgentAddress(rank=rank, size=size)
        self.context = AgentContext(
            event_loop=self.event_loop, rank=rank, size=size, address=address
        )

    def close(self):
        if self.context:
            self.context.close()
        self.event_loop.close()


class AgentContext:
    def __init__(
        self, *, event_loop: EventLoop, rank: int, size: int, address: SocketAddress
    ):
        self.rank = rank
        self.size = size
        self.address = address
        self.pairs: Dict[int, "Pair"] = {}
        self._event_loop = event_loop

    def getPair(self, peer_rank):
        if peer_rank not in self.pairs:
            self.pairs[peer_rank] = self.createPair(peer_rank)
        return self.pairs[peer_rank]

    def createPair(self, peer_rank):
        pair_address = self.address
        pair_address.addr = (pair_address.addr[0], pair_address.addr[1] + peer_rank)
        pair = Pair(
            event_loop=self._event_loop,
            self_rank=self.rank,
            peer_rank=peer_rank,
            address=pair_address,
        )
        self.pairs[peer_rank] = pair
        return pair

    def close(self):
        for peer_rank, pair in self.pairs.items():
            pair.close()
        self.pairs = {}

    def connectFull(self, store):
        def full_neighbor_fn(self_rank, peer_rank, size):
            return True

        return self._connectGivenNeighborFunc(store, full_neighbor_fn)

    def connectRing(self, store):
        def ring_neighbor_fn(self_rank, peer_rank, size):
            if peer_rank == (self_rank + 1) % size:
                return True
            if peer_rank == (self_rank - 1) % size:
                return True
            return False

        return self._connectGivenNeighborFunc(store, ring_neighbor_fn)

    def _connectGivenNeighborFunc(self, store, neighbor_fn):
        # It store the self address listening for the other ranks.
        self._all_address = []

        for i in range(self.size):
            is_neighbor = neighbor_fn(self.rank, i, self.size)
            if i == self.rank or not is_neighbor:
                # Just placeholder
                self._all_address.append(None)
                continue
            pair = self.createPair(i)
            self._all_address.append(pair.self_address)

        logger.debug("start listening done")

        # we need a global store to share between all processes.
        store.set(f"rank_addr_{self.rank}", self._all_address)

        # Connect others
        for i in range(self.size):
            is_neighbor = neighbor_fn(self.rank, i, self.size)
            if i == self.rank or not is_neighbor:
                continue
            # Other's file store
            others_all_address = store.get(f"rank_addr_{i}")
            # The listening address open for self.
            addr = others_all_address[self.rank]
            logger.debug(f"{self.rank} connect to {i}, addr {addr}")

            pair = self.getPair(i)
            pair.connect(addr)

        logger.debug("connect pairs done")
