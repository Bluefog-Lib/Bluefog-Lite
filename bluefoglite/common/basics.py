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
import io
import json
import os
import pickle
from typing import Any, Dict, Optional
from concurrent.futures import Executor, Future, ThreadPoolExecutor

import networkx as nx
import numpy as np

from bluefoglite.common import const
from bluefoglite.common.collective_comm import allreduce_tree
from bluefoglite.common.collective_comm import broadcast_one_to_all, broadcast_spreading
from bluefoglite.common.collective_comm import neighbor_allreduce
from bluefoglite.common.logger import Logger
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.common.tcp.buffer import (
    NumpyBuffer,
    SpecifiedBuffer,
    UnspecifiedBuffer,
)
from bluefoglite.common.store import FileStore
from bluefoglite.common.topology import GetRecvWeights


def _json_encode(obj: Any, encoding: str) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode(encoding)


def _json_decode(json_bytes: bytes, encoding: str) -> Dict:
    tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
    obj = json.load(tiow)
    tiow.close()
    return obj


def _maybe_convert_to_numeric(array: np.ndarray, inplace=False) -> np.ndarray:
    if array.dtype not in [np.float16, np.float32, np.float64, np.complex64]:
        if inplace:
            raise ValueError(
                "Cannot do inplace averge in the allreduce when input array "
                f"(dtype: {array.dtype})is not floating type."
            )
        Logger.get().warning(
            "The operation with average reduce should call the array with floating datatype "
            "but the provided one is %s. Auto-casting it.",
            array.dtype,
        )
        if array.dtype in [np.int8, np.int16]:
            array = array.astype(np.float16)
        elif array.dtype == np.int32:
            array = array.astype(np.float32)
        else:
            array = array.astype(np.float64)
    return array


@dataclasses.dataclass
class TopologyAndWeights:
    topology: nx.DiGraph
    default_self_weight: float
    default_src_weights: Dict[int, float]
    default_dst_weights: Dict[int, float]


class BlueFogLiteGroup:
    def __init__(self) -> None:
        self._agent: Optional[Agent] = None
        self._store: Optional[FileStore] = None

        self._rank: Optional[int] = None
        self._size: Optional[int] = None
        self._executor: Executor = ThreadPoolExecutor(
            max_workers=const.MAX_THREAD_POOL_WORKER
        )
        self._topology_and_weights: Optional[TopologyAndWeights] = None

    def init(self, store=None, rank: Optional[int] = None, size: Optional[int] = None):
        if store is None:
            _file_store_loc = os.getenv(const.BFL_FILE_STORE)
            if _file_store_loc is None:
                raise RuntimeError(
                    f"Environment variable {const.BFL_FILE_STORE} -- the path for"
                    "a writtable directory like tmp/.bluefoglite, must be provided."
                )
            self._store = FileStore(_file_store_loc)
        else:
            self._store = store

        _world_rank_env = os.getenv(const.BFL_WORLD_RANK)
        _world_size_env = os.getenv(const.BFL_WORLD_SIZE)
        if _world_rank_env is None or _world_size_env is None:
            raise RuntimeError(
                "All BluefogLite processes must be specified with "
                f"environment variable {const.BFL_WORLD_RANK} and "
                f"{const.BFL_WORLD_SIZE} with string value for integer."
            )
        self._rank = int(_world_rank_env) if rank is None else rank
        self._size = int(_world_size_env) if size is None else size

        self._agent = Agent()
        context = self._agent.createContext(rank=self._rank, size=self._size)
        context.connectFull(store=self._store)

    def shutdown(self):
        if self._agent is not None:
            self._agent.close()
        self._executor.shutdown()

    def rank(self) -> int:
        if self._rank is None:
            raise RuntimeError("Bluefoglite must call init() function first.")
        return self._rank

    def size(self) -> int:
        if self._size is None:
            raise RuntimeError("Bluefoglite must call init() function first.")
        return self._size

    def _check_rank(self, rank):
        error_msg = "dst or src must be an interger between 0 and size-1."
        assert isinstance(rank, int), error_msg
        assert rank >= 0, error_msg
        assert rank < self.size(), error_msg

    def set_topology(self, topology: nx.DiGraph) -> bool:
        """A function that sets the virtual topology MPI used.

        Args:
          Topo: A networkx.DiGraph object to decide the topology.

        Returns:
            A boolean value that whether topology is set correctly or not.
        """
        if not isinstance(topology, nx.DiGraph):
            raise TypeError("topology must be a networkx.DiGraph obejct.")
        if topology.number_of_nodes() != self.size():
            raise TypeError(
                "topology must be a networkx.DiGraph obejct with same number of "
                "nodes as bfl.size()."
            )
        _default_self_weight, _default_src_weights = GetRecvWeights(
            topology, self.rank()
        )
        _default_dst_weights = {
            int(r): 1.0 for r in topology.successors(self.rank()) if r != self.rank()
        }
        self._topology_and_weights = TopologyAndWeights(
            topology=topology.copy(),
            default_self_weight=_default_self_weight,
            default_src_weights=_default_src_weights,
            default_dst_weights=_default_dst_weights,
        )
        return True

    def send(self, dst, obj_or_array, tag=0):
        self._check_rank(dst)

        # TODO check send/recv type?
        if isinstance(obj_or_array, np.ndarray):
            buf = NumpyBuffer(self._agent.context, obj_or_array)
        else:
            message = pickle.dumps(obj_or_array)
            buf = SpecifiedBuffer(
                self._agent.context, memoryview(message), len(message)
            )

        # TODO: Check if dst is neighbor or not
        buf.send(dst)

    def recv(self, src, obj_or_array=None, tag=0):
        self._check_rank(src)

        # TODO check send/recv type?
        if isinstance(obj_or_array, np.ndarray):
            buf = NumpyBuffer(self._agent.context, obj_or_array)
            # TODO: Check if src is neighbor or not
            buf.recv(src)
            return obj_or_array

        ubuf = UnspecifiedBuffer(self._agent.context)
        ubuf.recv(src)
        obj = pickle.loads(ubuf.data)
        return obj

    def _prepare_numpy_buffer(self, array: np.ndarray) -> NumpyBuffer:
        if not isinstance(array, np.ndarray):
            raise ValueError("Input array has to be numpy array only for now")

        if self._agent is None or self._agent.context is None:
            raise RuntimeError(
                "Bluefoglite is not initialized. Forget to call bfl.init()?"
            )
        buf = NumpyBuffer(self._agent.context, array)
        return buf

    # TODO 1.Distinguish inplace versus not inplace change.
    # TODO 2. Add nonblocking version.
    def broadcast(
        self, *, array: np.ndarray, root_rank: int, inplace: bool
    ) -> np.ndarray:
        self._check_rank(root_rank)
        buf = self._prepare_numpy_buffer(array)
        out_buf = buf if inplace else buf.clone()

        if self.size() <= 4:
            broadcast_one_to_all(out_buf, root_rank, out_buf.context)
        else:
            broadcast_spreading(out_buf, root_rank, out_buf.context)
        return out_buf.array

    def broadcast_nonblocking(
        self, *, array: np.ndarray, root_rank: int, inplace: bool
    ) -> Future:
        return self._executor.submit(
            self.broadcast, array=array, root_rank=root_rank, inplace=inplace
        )

    def allreduce(self, *, array: np.ndarray, agg_op: str, inplace: bool) -> np.ndarray:
        if agg_op == "AVG":
            array = _maybe_convert_to_numeric(array, inplace=inplace)
        buf = self._prepare_numpy_buffer(array)
        out_buf = buf if inplace else buf.clone()

        allreduce_tree(out_buf, out_buf.context, agg_op=agg_op)
        return out_buf.array

    def allreduce_nonblocking(
        self, *, array: np.ndarray, agg_op: str, inplace: bool
    ) -> Future:
        return self._executor.submit(
            self.allreduce, array=array, agg_op=agg_op, inplace=inplace
        )

    def neighbor_allreduce(
        self,
        *,
        array: np.ndarray,
        self_weight: Optional[float],
        src_weights: Optional[Dict[int, float]],
        dst_weights: Optional[Dict[int, float]],
    ) -> np.ndarray:
        # TODO 1. add topology check service.
        if (
            self_weight is None
            and src_weights is None
            and dst_weights is None
            and (self._topology_and_weights is not None)
        ):
            self_weight = self._topology_and_weights.default_self_weight
            src_weights = self._topology_and_weights.default_src_weights
            dst_weights = self._topology_and_weights.default_dst_weights

        if self_weight is None or src_weights is None or dst_weights is None:
            raise ValueError(
                "Must provide all self_weight, src_weights, and dst_weights "
                "arguments or set static topology."
            )

        array = _maybe_convert_to_numeric(array, inplace=False)
        in_buf = self._prepare_numpy_buffer(array)
        out_buf = in_buf.clone()
        neighbor_allreduce(
            in_buf=in_buf,
            out_buf=out_buf,
            self_weight=self_weight,
            src_weights=src_weights,
            dst_weights=dst_weights,
        )
        return out_buf.array

    def neighbor_allreduce_nonblocking(
        self,
        *,
        array: np.ndarray,
        self_weight: float,
        src_weights: Dict[int, float],
        dst_weights: Dict[int, float],
    ) -> Future:
        return self._executor.submit(
            self.neighbor_allreduce,
            array=array,
            self_weight=self_weight,
            src_weights=src_weights,
            dst_weights=dst_weights,
        )
