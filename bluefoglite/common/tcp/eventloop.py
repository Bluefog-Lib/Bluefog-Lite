import abc
import functools
import select
import selectors
import socket
import threading
from typing import Callable, Dict, List, Union
from bluefoglite.common.logger import logger


class Handler(abc.ABC):

    @abc.abstractmethod
    def handleEvent(self, event: int):
        raise NotImplementedError


# TODO(ybc) make this class singleton?
class EventLoop(object):

    def __init__(self):
        self.sel = selectors.DefaultSelector()
        self.done = False  # Python assignment to simple variable is "atomic"
        self.running_thread = None

        self._cv = threading.Condition()

    def __del__(self):
        self.close()

    def run(self):
        if self.running_thread is not None:
            print('Event Loop is already running')
            return

        # print('start running loop')
        self.running_thread = threading.Thread(
            target=EventLoop._run, args=(self,))
        self.running_thread.start()

    def _run(self):
        while not self.done:
            # self._cv.notify_all()

            # Find a better timeout choice? for closing the loop.
            events_list = self.sel.select(2)  # timeout after 2 seconds
            for key, event in events_list:
                event_ = 'r' if event == 1 else 'w' if event == 2 else 'rw'
                logger.debug("%s %s", event_, key.fileobj)
                # key is the SelectorKey instance corresponding to a ready file object.
                # SelectorKey is a namedtuple: (fileobj, fd, events, data)
                # We force the data to be the instance of abstract class Handler.
                key.data.handleEvent(event)
        # print('end running loop')

    def register(self, fd: Union[int, socket.socket], event: int, handler: Handler):
        self.sel.register(fd, event, handler)

    def modify(self, fd: Union[int, socket.socket], event: int, handler: Handler):
        self.sel.modify(fd, event, handler)

    def unregister(self, fd: Union[int, socket.socket]):
        self.sel.unregister(fd)
        # # make sure `unregister` returned after the loop ticked.
        # self._cv.acquire()
        # self._cv.wait()
        # self._cv.release()

    def close(self):
        self.done = True
        self.running_thread.join()

        self.sel.close()
