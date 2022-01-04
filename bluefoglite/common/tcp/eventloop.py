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
import selectors
import socket
import threading
from typing import Union
from bluefoglite.common.logger import logger


class Handler(abc.ABC):
    @abc.abstractmethod
    def handleEvent(self, event: int):
        raise NotImplementedError


# TODO(ybc) make this class singleton?
class EventLoop:
    def __init__(self):
        self.sel = selectors.DefaultSelector()
        self.done = False  # Python assignment to simple variable is "atomic"
        self.running_thread = None

        self._cv = threading.Condition()

    def __del__(self):
        self.close()

    def run(self):
        if self.running_thread is not None:
            print("Event Loop is already running")
            return

        # print('start running loop')
        self.running_thread = threading.Thread(target=EventLoop._run, args=(self,))
        self.running_thread.start()

    def _run(self):
        while not self.done:
            # self._cv.notify_all()

            # Find a better timeout choice? for closing the loop.
            events_list = self.sel.select(2)  # timeout after 2 seconds
            for key, event in events_list:
                # TODO Add error handling

                # key is the SelectorKey instance corresponding to a ready file object.
                # SelectorKey is a namedtuple: (fileobj, fd, events, data)
                # We force the data to be the instance of abstract class Handler.
                key.data.handleEvent(event)

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
