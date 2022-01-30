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

import numpy as np  # type: ignore

from bluefoglite.common import const
from bluefoglite.common.tcp.agent import Agent
from bluefoglite.common.tcp.buffer import SpecifiedBuffer, UnspecifiedBuffer
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
        context.connectRing(store=self._store)

    def shutdown(self):
        if self._agent is not None:
            self._agent.close()

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
            buf = SpecifiedBuffer(
                self._agent.context, obj_or_array.data, obj_or_array.nbytes
            )
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
            buf = SpecifiedBuffer(
                self._agent.context, obj_or_array.data, obj_or_array.nbytes
            )
            # TODO: Check if src is neighbor or not
            buf.recv(src)
            return obj_or_array

        ubuf = UnspecifiedBuffer(self._agent.context)
        ubuf.recv(src)
        obj = pickle.loads(ubuf.data)
        return obj
