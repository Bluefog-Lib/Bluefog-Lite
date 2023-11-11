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

from typing import Dict, Optional

import networkx as nx
import torch
import torch.distributed as dist

from bluefoglite.common.torch_backend import AsyncWork, BlueFogLiteGroup, ReduceOp

_global_group = BlueFogLiteGroup()

__all__ = [
    "ReduceOp",
    "init",
    "shutdown",
    "size",
    "rank",
    "send",
    "recv",
    "isend",
    "irecv",
    "set_topology",
    "neighbor_allreduce",
    "neighbor_allreduce_nonblocking",
    "broadcast_nonblocking",
    "broadcast",
    "allreduce_nonblocking",
    "allreduce",
]
# import basic methods and wrap it with default global group.


def init(backend: str = "gloo", *, group=None):
    if group is None:
        group = _global_group
    group.init(backend=backend)


def shutdown(group=None):
    if group is None:
        group = _global_group
    group.shutdown()


def size(group=None) -> int:
    if group is None:
        group = _global_group
    return group.size()


def rank(group=None) -> int:
    if group is None:
        group = _global_group
    return group.rank()


def send(tensor, dst, *, tag: int = 0, group=None) -> None:
    if group is None:
        group = _global_group
    group.send(tensor=tensor, dst=dst, tag=tag)


def recv(tensor, src, *, tag: int = 0, group=None) -> None:
    if group is None:
        group = _global_group
    group.recv(tensor=tensor, src=src, tag=tag)


def isend(tensor, dst, *, tag: int = 0, group=None) -> dist.Work:
    if group is None:
        group = _global_group
    return group.send(tensor=tensor, dst=dst, tag=tag)


def irecv(tensor, src, *, tag: int = 0, group=None) -> dist.Work:
    if group is None:
        group = _global_group
    return group.recv(tensor=tensor, src=src, tag=tag)


def set_topology(topology: nx.DiGraph, *, group=None):
    if group is None:
        group = _global_group
    return group.set_topology(topology=topology)


def neighbor_allreduce(
    tensor: torch.Tensor,
    *,
    self_weight: Optional[float] = None,
    src_weights: Optional[Dict[int, float]] = None,
    dst_weights: Optional[Dict[int, float]] = None,
    inplace: bool = False,
    group=None,
) -> torch.Tensor:
    if group is None:
        group = _global_group
    return group.neighbor_allreduce(
        tensor=tensor,
        self_weight=self_weight,
        src_weights=src_weights,
        dst_weights=dst_weights,
        inplace=inplace,
    )


def neighbor_allreduce_nonblocking(
    tensor: torch.Tensor,
    *,
    self_weight: Optional[float] = None,
    src_weights: Optional[Dict[int, float]] = None,
    dst_weights: Optional[Dict[int, float]] = None,
    inplace: bool = False,
    group=None,
) -> AsyncWork:
    if group is None:
        group = _global_group
    return group.neighbor_allreduce_nonblocking(
        tensor=tensor,
        self_weight=self_weight,
        src_weights=src_weights,
        dst_weights=dst_weights,
        inplace=inplace,
    )


def broadcast(
    tensor: torch.Tensor, root_rank: int, *, inplace: bool = False, group=None
) -> torch.Tensor:
    if group is None:
        group = _global_group
    return group.broadcast(tensor=tensor, root_rank=root_rank, inplace=inplace)


def broadcast_nonblocking(
    tensor: torch.Tensor, root_rank: int, *, inplace: bool = False, group=None
) -> AsyncWork:
    if group is None:
        group = _global_group
    return group.broadcast_nonblocking(
        tensor=tensor, root_rank=root_rank, inplace=inplace
    )


def allreduce(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.AVG,
    *,
    inplace: bool = False,
    group=None,
) -> torch.Tensor:
    if group is None:
        group = _global_group
    return group.allreduce(tensor=tensor, op=op, inplace=inplace)


def allreduce_nonblocking(
    tensor: torch.Tensor,
    op: ReduceOp = ReduceOp.AVG,
    *,
    inplace: bool = False,
    group=None,
) -> AsyncWork:
    if group is None:
        group = _global_group
    return group.allreduce_nonblocking(tensor=tensor, op=op, inplace=inplace)


def barrier():
    allreduce(torch.Tensor([1.0]))
