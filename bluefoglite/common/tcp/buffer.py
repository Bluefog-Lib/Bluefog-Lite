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


import abc
import threading
from typing import Optional, TYPE_CHECKING

import numpy as np

from bluefoglite.common.handle_manager import HandleManager, EventStatus

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from bluefoglite.common.tcp.agent import AgentContext


class Buffer(abc.ABC):
    def __init__(self) -> None:
        self.data = b""
        self.buffer_view = memoryview(self.data)
        self.buffer_length = 0
        self.mutex = threading.Lock()
        self.cv = threading.Condition(self.mutex)
        self.hm = HandleManager.getInstance()

    def waitCompletion(self, handle: int, timeout=None):
        assert self.hm.wait(handle=handle, timeout=timeout)
        self.hm.release(handle=handle)

    def handleCompletion(self, handle: int, event_status: Optional[EventStatus] = None):
        self.hm.markDone(handle, event_status)

    @abc.abstractmethod
    def irecv(
        self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0
    ) -> int:
        return NotImplemented

    @abc.abstractmethod
    def recv(self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0):
        return NotImplemented


class SpecifiedBuffer(Buffer):
    def __init__(
        self, context: "AgentContext", buffer_view: memoryview, buffer_length: int
    ) -> None:
        super().__init__()
        self.context = context
        self.buffer_view = buffer_view
        self.buffer_length = buffer_length

    def isend(
        self, dst: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0
    ) -> int:
        if nbytes == -1:
            nbytes = self.buffer_length

        handle = self.hm.allocate()
        # TODO: Make some verificaiton here
        self.context.getOrCreatePair(dst).send(
            self, handle=handle, nbytes=nbytes, offset=offset, slot=slot
        )
        return handle

    def irecv(
        self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0
    ) -> int:
        if nbytes == -1:
            nbytes = self.buffer_length - offset

        handle = self.hm.allocate()
        # TODO Make some verificaiton here
        self.context.getOrCreatePair(src).recv(
            self, handle=handle, nbytes=nbytes, offset=offset, slot=slot
        )
        return handle

    def send(self, dst: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0):
        handle = self.isend(dst, nbytes=nbytes, offset=offset, slot=slot)
        self.waitCompletion(handle)

    def recv(self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0):
        handle = self.irecv(src, nbytes=nbytes, offset=offset, slot=slot)
        self.waitCompletion(handle)


class NumpyBuffer(SpecifiedBuffer):
    def __init__(self, context: "AgentContext", array: np.ndarray) -> None:
        super().__init__(context, array.data, array.nbytes)

        self._array = array  # Should not use it since it may change?

        self.dtype = array.dtype
        self.shape = array.shape
        self.itemsize = array.itemsize

    def clone(self):
        new_array = np.empty(self.shape, dtype=self.dtype)
        return NumpyBuffer(self.context, new_array)


class UnspecifiedBuffer(Buffer):
    def __init__(self, context: "AgentContext"):
        super().__init__()
        self.context = context
        self.data = b""

    def irecv(
        self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0
    ) -> int:
        handle = self.hm.allocate()
        # TODO Make some verificaiton here
        self.context.getOrCreatePair(src).recv(
            self, handle=handle, nbytes=-1, offset=0, slot=0
        )
        return handle

    def recv(self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0):
        handle = self.irecv(src)
        self.waitCompletion(handle)
