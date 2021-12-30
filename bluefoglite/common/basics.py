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
from typing import Any, Dict, Optional

from bluefoglite.common.tcp.agent import Agent
from bluefoglite.common.tcp.buffer import Buffer
from bluefoglite.common.store import FileStore
from bluefoglite.common.logger import logger

import numpy as np  # type: ignore


def _json_encode(obj: Any, encoding: str) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode(encoding)


def _json_decode(json_bytes: bytes, encoding: str) -> Dict:
    tiow = io.TextIOWrapper(io.BytesIO(json_bytes), encoding=encoding, newline="")
    obj = json.load(tiow)
    tiow.close()
    return obj


class BlueFogLiteGroup(object):
    def __init__(self) -> None:
        self._agent = None

    def init(self, store=None):
        self._agent = Agent()
        self._agent.createContext(rank=self.rank(), size=self.size())
        self._store = FileStore(os.getenv("BFL_FILE_STORE")) if store is None else store

        self._agent.context.connectRing(store=self._store)

    def shutdown(self):
        if self._agent is not None:
            self._agent.close()

    def rank(self):
        rank = os.getenv("BFL_WORLD_RANK")
        return int(rank) if rank else 0

    def size(self):
        size = os.getenv("BFL_WORLD_SIZE")
        return int(size) if size else 1

    def _check_rank(self, rank):
        error_msg = "dst or src must be an interger between 0 and size-1."
        assert isinstance(rank, int), error_msg
        assert rank >= 0, error_msg
        assert rank < self.size(), error_msg

    def send(self, dst, obj_or_array, tag=0):
        self._check_rank(dst)

        # TODO check send/recv type?
        if isinstance(obj_or_array, np.ndarray):
            buf = Buffer(self._agent.context, obj_or_array.data, obj_or_array.nbytes)
        else:
            message = _json_encode(obj_or_array, "utf-8")
            buf = Buffer(self._agent.context, memoryview(message), len(message))

        # TODO: Check if dst is neighbor or not
        buf.send(dst)

    def recv(self, src, obj_or_array=None, tag=0):
        self._check_rank(src)

        # TODO check send/recv type?
        if isinstance(obj_or_array, np.ndarray):
            buf = Buffer(self._agent.context, obj_or_array.data, obj_or_array.nbytes)
        else:
            raise NotImplemented
            buf = Buffer(self._agent.context, memoryview(message), len(message))

        # TODO: Check if src is neighbor or not
        buf.recv(src)
        return obj_or_array
