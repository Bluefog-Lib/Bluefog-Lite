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

import io
import json
import os
import pickle
from typing import Any, Dict, Optional
from concurrent.futures import Executor, Future, ThreadPoolExecutor

import numpy as np  # type: ignore

from bluefoglite.common import const
from bluefoglite.common.collective_comm import allreduce_tree
from bluefoglite.common.collective_comm import broadcast_one_to_all, broadcast_spreading
from bluefoglite.common.logger import Logger
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.common.tcp.buffer import (
    NumpyBuffer,
    SpecifiedBuffer,
    UnspecifiedBuffer,
)
from bluefoglite.common.store import FileStore


def _json_encode(obj: Any, encoding: str) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode(encoding)


def _json_decode(json_bytes: bytes, encoding: str) -> Dict:
    tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
    obj = json.load(tiow)
    tiow.close()
    return obj


class BlueFogLiteGroup:
    def __init__(self) -> None:
        self._agent: Optional[Agent] = None
        self._store: Optional[FileStore] = None

        self._rank: Optional[int] = None
        self._size: Optional[int] = None

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
            store = store

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

        self._executor: Executor = ThreadPoolExecutor(max_workers=4)

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
        if agg_op == "AVG" and array.dtype not in [
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
        ]:
            Logger.get().warning(
                "allreduce with AVG should call the array with floating datatype "
                "but the provided one is %s. Auto-casting it.",
                array.dtype,
            )
            if array.dtype in [np.int8, np.int16]:
                array = array.astype(np.float16)
            elif array.dtype == np.int32:
                array = array.astype(np.float32)
            else:
                array = array.astype(np.float64)

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
