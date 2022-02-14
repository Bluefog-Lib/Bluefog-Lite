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

from typing import Any, Dict
from concurrent.futures import Future
import numpy as np  # type: ignore

from bluefoglite.common.basics import BlueFogLiteGroup

_global_group = BlueFogLiteGroup()

# import basic methods and wrap it with default global group.


def init(group=None):
    if group is None:
        group = _global_group
    group.init()


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


def send(dst, obj_or_array: Any, *, tag: int = 0, group=None):
    if group is None:
        group = _global_group
    group.send(dst=dst, obj_or_array=obj_or_array, tag=tag)


def recv(src, obj_or_array: Any, *, tag: int = 0, group=None):
    if group is None:
        group = _global_group
    return group.recv(src=src, obj_or_array=obj_or_array, tag=tag)


def broadcast(
    array: np.ndarray,
    root_rank: int,
    *,
    inplace: bool = False,
    tag: int = 0,
    group=None
):
    if group is None:
        group = _global_group
    return group.broadcast(array=array, root_rank=root_rank, inplace=inplace)


def broadcast_nonblocking(
    array: np.ndarray, root_rank: int, *, inplace=False, tag=0, group=None
) -> Future:
    if group is None:
        group = _global_group
    return group.broadcast_nonblocking(
        array=array, root_rank=root_rank, inplace=inplace
    )


def allreduce(
    array: np.ndarray,
    *,
    agg_op: str = "AVG",
    inplace: bool = False,
    tag: int = 0,
    group=None
):
    if group is None:
        group = _global_group
    return group.allreduce(array=array, agg_op=agg_op, inplace=inplace)


def allreduce_nonblocking(
    array: np.ndarray,
    *,
    agg_op: str = "AVG",
    inplace: bool = False,
    tag: int = 0,
    group=None
) -> Future:
    if group is None:
        group = _global_group
    return group.allreduce_nonblocking(array=array, agg_op=agg_op, inplace=inplace)


def neighbor_allreduce(
    array: np.ndarray,
    *,
    self_weight: float,
    src_weights: Dict[int, float],
    dst_weights: Dict[int, float],
    group=None
) -> np.ndarray:
    if group is None:
        group = _global_group
    return group.neighbor_allreduce(
        array=array,
        self_weight=self_weight,
        src_weights=src_weights,
        dst_weights=dst_weights,
    )


def neighbor_allreduce_nonblocking(
    array: np.ndarray,
    *,
    self_weight: float,
    src_weights: Dict[int, float],
    dst_weights: Dict[int, float],
    group=None
) -> Future:
    if group is None:
        group = _global_group
    return group.neighbor_allreduce_nonblocking(
        array=array,
        self_weight=self_weight,
        src_weights=src_weights,
        dst_weights=dst_weights,
    )