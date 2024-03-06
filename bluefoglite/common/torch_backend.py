# Copyright 2023 Bluefog Team. All Rights Reserved.
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
from collections.abc import Iterable
import functools
from enum import Enum
import os
from typing import Any, Dict, List, Tuple, Optional, Union, Callable

import networkx as nx
import torch
import torch.distributed as dist

from bluefoglite.common import const
from bluefoglite.common.topology import GetRecvWeights


@dataclasses.dataclass
class TopologyAndWeights:
    topology: nx.DiGraph
    default_self_weight: float
    default_src_weights: Dict[int, float]
    default_dst_weights: Dict[int, float]


class ReduceOp(Enum):
    AVG = 0
    SUM = (dist.ReduceOp.SUM,)
    PRODUCT = (dist.ReduceOp.PRODUCT,)
    MIN = (dist.ReduceOp.MIN,)
    MAX = (dist.ReduceOp.MAX,)
    BAND = (dist.ReduceOp.BAND,)
    BOR = (dist.ReduceOp.BOR,)
    BXOR = (dist.ReduceOp.BXOR,)
    PREMUL_SUM = (dist.ReduceOp.PREMUL_SUM,)


class AsyncWork:
    def __init__(
        self,
        work: Union[dist.Work, List[dist.Work]],
        post_func: Optional[Callable] = None,
    ):
        self._work = work
        self._post_func = post_func

    def wait(self) -> Optional[Any]:
        if isinstance(self._work, Iterable):
            for w in self._work:
                w.wait()
        else:
            self._work.wait()
        if self._post_func:
            return self._post_func()
        return None


class BlueFogLiteGroup:
    def __init__(self) -> None:
        self._rank: Optional[int] = None
        self._size: Optional[int] = None
        self._topology_and_weights: Optional[TopologyAndWeights] = None
        self._process_group: Optional[dist.ProcessGroup] = None

    @property
    def process_group(self):
        if not self._process_group:
            raise RuntimeError("Initialize the Bluefoglite first.")
        return self._process_group

    def is_initialized(self) -> bool:
        return self._process_group is not None

    def init_from_process_group(self, process_group: dist.ProcessGroup):
        assert isinstance(process_group, dist.ProcessGroup)
        self._process_group = process_group

    def init(
        self, backend="gloo", rank: Optional[int] = None, size: Optional[int] = None
    ):
        if not dist.is_available():
            raise EnvironmentError(
                "Please install torch with distributed package support."
            )
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
        self._backend = "undefined" if backend is None else backend

        dist.init_process_group(
            backend=backend,
            world_size=self._size,
            rank=self._rank,
            group_name="bluefog-lite-global",
        )
        self._process_group = dist.GroupMember.WORLD
        self._rank = dist.get_rank()
        self._size = dist.get_world_size()

    def shutdown(self):
        if self._process_group is not None:
            dist.destroy_process_group(self._process_group)
            self._process_group = None

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

    def load_topology(self, group=None):
        return self._topology_and_weights.topology

    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> AsyncWork:
        self._check_rank(dst)
        return AsyncWork(work=self.process_group.send([tensor], dstRank=dst, tag=tag))

    def irecv(self, tensor: torch.Tensor, src: int, tag: int = 0) -> AsyncWork:
        self._check_rank(src)
        return AsyncWork(work=self.process_group.recv([tensor], srcRank=src, tag=tag))

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        self.isend(tensor=tensor, dst=dst, tag=tag).wait()

    def recv(self, tensor: torch.Tensor, src: int, tag: int = 0) -> None:
        self.irecv(tensor=tensor, src=src, tag=tag).wait()

    def wait(self, work: AsyncWork) -> Any:
        return work.wait()

    def neighbor_allreduce_nonblocking(
        self,
        tensor: torch.Tensor,
        *,
        self_weight: Optional[float],
        src_weights: Optional[Dict[int, float]],
        dst_weights: Optional[Dict[int, float]],
        inplace: bool = False,
    ) -> AsyncWork:
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

        op_list = []
        for dst, weight in dst_weights.items():
            comm_tensor = (
                tensor.mul(weight).to("cpu")
                if self._backend == "gloo" and tensor.device.type != "cpu"
                else tensor.mul(weight)
            )
            op_list.append(
                dist.P2POp(
                    dist.isend,
                    comm_tensor,
                    peer=dst,
                    group=self.process_group,
                )
            )
        src_weights_items = list(src_weights.items())
        tmp_recv_tensors_concat = torch.zeros(
            len(src_weights_items),
            *tensor.shape,
            device=(
                "cpu"
                if self._backend == "gloo" and tensor.device.type != "cpu"
                else tensor.device
            ),
        )
        for idx, (src, _) in enumerate(src_weights_items):
            op_list.append(
                dist.P2POp(
                    dist.irecv,
                    tmp_recv_tensors_concat[idx],
                    peer=src,
                    group=self.process_group,
                )
            )

        reqs = dist.batch_isend_irecv(op_list)

        def post_func(
            tensor: torch.Tensor,
            tmp_recv_tensors_concat: torch.Tensor,
            self_weight: float,
            src_weights_items: List[Tuple[int, float]],
        ) -> torch.Tensor:
            tensor_ = tensor if inplace else tensor.detach().clone()
            tensor_.mul_(self_weight)
            # move tmp_recv_tensors_concat from cpu to cuda.
            if self._backend == "gloo" and tensor.device.type != "cpu":
                tmp_recv_tensors_concat = tmp_recv_tensors_concat.to(tensor_.device)
            for idx, (_, weight) in enumerate(src_weights_items):
                tensor_.add_(tmp_recv_tensors_concat[idx], alpha=weight)
            del tmp_recv_tensors_concat
            return tensor_

        return AsyncWork(
            reqs,
            functools.partial(
                post_func,
                tensor=tensor,
                tmp_recv_tensors_concat=tmp_recv_tensors_concat,
                self_weight=self_weight,
                src_weights_items=src_weights_items,
            ),
        )

    def neighbor_allreduce(
        self,
        tensor: torch.Tensor,
        *,
        self_weight: Optional[float],
        src_weights: Optional[Dict[int, float]],
        dst_weights: Optional[Dict[int, float]],
        inplace: bool = False,
    ) -> torch.Tensor:
        return self.neighbor_allreduce_nonblocking(
            tensor=tensor,
            self_weight=self_weight,
            src_weights=src_weights,
            dst_weights=dst_weights,
            inplace=inplace,
        ).wait()

    def broadcast_nonblocking(
        self, tensor: torch.Tensor, root_rank: int, inplace: bool = True
    ) -> AsyncWork:
        opts = dist.BroadcastOptions()
        opts.rootRank = root_rank
        opts.rootTensor = 0
        if self.rank() == root_rank:
            _tensor = tensor
        else:
            _tensor = tensor if inplace else tensor.detach().clone()

        def post_func(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        return AsyncWork(
            self.process_group.broadcast([_tensor], opts),
            functools.partial(post_func, tensor=_tensor),
        )

    def broadcast(
        self, tensor: torch.Tensor, root_rank: int, inplace: bool = True
    ) -> torch.Tensor:
        return self.broadcast_nonblocking(
            tensor=tensor, root_rank=root_rank, inplace=inplace
        ).wait()

    def allreduce_nonblocking(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.AVG,
        inplace: bool = True,
    ) -> AsyncWork:
        opts = dist.AllreduceOptions()
        opts.reduceOp = op.value[0] if op != ReduceOp.AVG else dist.ReduceOp.SUM
        _tensor = tensor if inplace else tensor.detach().clone()

        def post_func(tensor: torch.Tensor, op: ReduceOp, size: int) -> torch.Tensor:
            return tensor.mul_(1 / size) if op == ReduceOp.AVG else tensor

        return AsyncWork(
            self.process_group.allreduce([_tensor], opts),
            functools.partial(post_func, tensor=_tensor, op=op, size=self.size()),
        )

    def allreduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.AVG,
        inplace: bool = True,
    ) -> torch.Tensor:
        return self.allreduce_nonblocking(tensor=tensor, op=op, inplace=inplace).wait()
