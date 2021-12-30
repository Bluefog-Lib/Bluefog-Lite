import threading
from typing import TYPE_CHECKING

from bluefoglite.common.handle_manager import HandleManager

if TYPE_CHECKING:
    from bluefoglite.common.tcp.agent import AgentContext

class Buffer:

    def __init__(self, context: 'AgentContext', buffer_view: memoryview, buffer_length: int) -> None:
        self.context = context
        self.buffer_view = buffer_view
        self.buffer_length = buffer_length

        self.hm = HandleManager.getInstance()

        self.mutex = threading.Lock()
        self.cv = threading.Condition(self.mutex)

    def isend(self, dst: int, *,  nbytes: int = -1, offset: int = 0, slot: int = 0) -> int:
        if nbytes == -1:
            nbytes = self.buffer_length

        handle = self.hm.allocate()
        # TODO Make some verificaiton here
        self.context.getPair(dst).send(
            self, handle=handle, nbytes=nbytes, offset=offset, slot=slot)
        return handle

    def irecv(self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0) -> int:
        if nbytes == -1:
            nbytes = self.buffer_length - offset

        handle = self.hm.allocate()
        # TODO Make some verificaiton here
        self.context.getPair(src).recv(
            self, handle=handle, nbytes=nbytes, offset=offset, slot=slot)
        return handle

    def send(self, dst: int, *,  nbytes: int = -1, offset: int = 0, slot: int = 0):
        handle = self.isend(dst, nbytes=nbytes,
                            offset=offset, slot=slot)
        self.waitCompletion(handle)

    def recv(self, src: int, *, nbytes: int = -1, offset: int = 0, slot: int = 0):
        handle = self.irecv(src, nbytes=nbytes,
                            offset=offset, slot=slot)
        self.waitCompletion(handle)

    def waitCompletion(self, handle: int, timeout=None):
        assert self.hm.wait(handle=handle, timeout=timeout)
        self.hm.release(handle=handle)

    def handleCompletion(self, handle):
        self.hm.markDone(handle)
