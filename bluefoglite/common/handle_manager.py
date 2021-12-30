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

import threading
from typing import Dict, Optional

# This should be a singleton class
class HandleManager:
    __instance: Optional["HandleManager"] = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if HandleManager.__instance == None:
            HandleManager.__instance = HandleManager()
        return HandleManager.__instance

    def __init__(self):
        if self.__instance is not None:
            raise RuntimeError(
                "HandleManager is singleton and " "should be used with getInstance()."
            )
        self.status: Dict[int, bool] = {}  # Handle -> Finished Or Not
        self._last_handle = -1  # should be atomic<int> but it is fine in python
        self.mutex = threading.Lock()
        self.cv = threading.Condition(self.mutex)

    @property
    def last_handle(self):
        return self._last_handle

    def allocate(self) -> int:
        with self.mutex:
            self._last_handle += 1
            self.status[self._last_handle] = False
            return self._last_handle

    def poll(self, handle: int) -> bool:
        with self.mutex:
            return self.status.get(handle, False)

    def markDone(self, handle: int):
        with self.mutex:
            self.status[handle] = True
            self.cv.notify_all()

    def release(self, handle: int) -> bool:
        with self.mutex:
            return self.status.pop(handle)

    def wait(self, handle: int, timeout: int = None) -> bool:
        with self.mutex:
            self.cv.wait_for(lambda: self.status[handle], timeout)
        return self.status[handle]

    def _reset(self):
        """This is a danger function and does not handle event notification."""
        with self.mutex:
            self.status = {}
            self._last_handle = -1
