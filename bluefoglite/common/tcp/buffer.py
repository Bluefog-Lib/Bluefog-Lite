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
from typing import Optional, TYPE_CHECKING, Tuple, Union

import numpy as np

from bluefoglite.common.handle_manager import HandleManager, EventStatus
from bluefoglite.common.util import numpy_to_bfl_dtype, bfl_to_numpy_dtype, TDtype

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from bluefoglite.common.tcp.agent import AgentContext


class Buffer(abc.ABC):  # pylint: disable=too-many-instance-attributes
    hm = HandleManager.getInstance()

    def __init__(self) -> None:
        self.data = b""
        self.buffer_view = memoryview(self.data)
        self.buffer_length = 0
        self.mutex = threading.Lock()
        self.cv = threading.Condition(self.mutex)

        # Only exists when we use numpy or similar style array
        self.shape: Optional[Tuple[int, ...]] = None
        self.ndim: Optional[int] = None
        self.dtype: Optional[TDtype] = None
        self.itemsize: Optional[int] = None

    @classmethod
    def waitCompletion(cls, handle: int, timeout=None):
        assert cls.hm.wait(handle=handle, timeout=timeout)
        cls.hm.release(handle=handle)

    @classmethod
    def handleCompletion(cls, handle: int, event_status: Optional[EventStatus] = None):
        cls.hm.markDone(handle, event_status)

    @abc.abstractmethod
    def irecv(self, src: int, *, nbytes: int = -1, offset: int = 0) -> int:
        return NotImplemented

    @abc.abstractmethod
    def recv(self, src: int, *, nbytes: int = -1, offset: int = 0):
        return NotImplemented


class SpecifiedBuffer(Buffer):
    def __init__(
        self, context: "AgentContext", buffer_view: memoryview, buffer_length: int
    ) -> None:
        super().__init__()
        self.context = context
        self.buffer_view = buffer_view
        self.buffer_length = buffer_length
        if self.buffer_view.format != "c":
            raise ValueError("The memoryview of buffer should be in char format.")
        if not self.buffer_view.contiguous:
            raise ValueError(
                "Only support the case that buffer_view has contiguous memeory."
            )

    def isend(self, dst: int, *, nbytes: int = -1, offset: int = 0) -> int:
        if nbytes == -1:
            nbytes = self.buffer_length

        handle = self.hm.allocate()
        # TODO: Make some verificaiton here
        self.context.getOrCreatePair(dst).send(
            self, handle=handle, nbytes=nbytes, offset=offset
        )
        return handle

    def irecv(self, src: int, *, nbytes: int = -1, offset: int = 0) -> int:
        if nbytes == -1:
            nbytes = self.buffer_length - offset

        handle = self.hm.allocate()
        # TODO Make some verificaiton here
        self.context.getOrCreatePair(src).recv(
            self, handle=handle, nbytes=nbytes, offset=offset
        )
        return handle

    def send(self, dst: int, *, nbytes: int = -1, offset: int = 0):
        handle = self.isend(dst, nbytes=nbytes, offset=offset)
        self.waitCompletion(handle)

    def recv(self, src: int, *, nbytes: int = -1, offset: int = 0):
        handle = self.irecv(src, nbytes=nbytes, offset=offset)
        self.waitCompletion(handle)


class NumpyBuffer(SpecifiedBuffer):
    def __init__(self, context: "AgentContext", array: np.ndarray) -> None:
        # .cast("c") is crucial for the downstream function because
        # everything index, slicing, etc will be operated in bytes only
        # and ignore the original itemsize, stride, shape, etc.
        super().__init__(context, array.data.cast("c"), array.nbytes)

        self.array = array  # Should not use it since it may change?

        # The logical structure of NumPy-style arrays is defined by
        #   itemsize, ndim, shape and strides.
        self.shape: Tuple[int, ...] = array.shape
        self.itemsize = array.itemsize
        self.ndim = array.ndim
        self.dtype = numpy_to_bfl_dtype(array.dtype)

    def create_new_buffer(self, shape: Tuple[int, ...]) -> "NumpyBuffer":
        if any(s <= 0 for s in shape):
            raise ValueError("shape should be the tuple with positive integer")
        empty_array = np.zeros(shape, dtype=bfl_to_numpy_dtype(self.dtype))
        return NumpyBuffer(self.context, array=empty_array)

    def clone(self):
        return NumpyBuffer(self.context, self.array.copy())

    # TODO: numerical ops should not be member function of Buffer.
    def add_(self, other_buf: "NumpyBuffer"):
        self.array += other_buf.array

    def mul_(self, factor: Union[int, float]):
        self.array *= factor

    def div_(self, factor: Union[int, float]):
        self.array /= factor


class UnspecifiedBuffer(Buffer):
    def __init__(self, context: "AgentContext"):
        super().__init__()
        self.context = context
        self.data = b""

    def irecv(self, src: int, *, nbytes: int = -1, offset: int = 0) -> int:
        handle = self.hm.allocate()
        # TODO Make some verificaiton here
        self.context.getOrCreatePair(src).recv(self, handle=handle, nbytes=-1, offset=0)
        return handle

    def recv(self, src: int, *, nbytes: int = -1, offset: int = 0):
        handle = self.irecv(src)
        self.waitCompletion(handle)
