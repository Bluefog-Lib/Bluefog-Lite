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

from dataclasses import dataclass
import enum
import threading
from typing import Dict, Optional


class EventStatusEnum(enum.Enum):
    UNKNOWN = 0
    ALLOCATED = 1
    DONE = 2
    ERROR = 3


@dataclass
class EventStatus:
    status: EventStatusEnum
    err: str


# This should be a singleton class
class HandleManager:
    __instance: Optional["HandleManager"] = None

    @staticmethod
    def getInstance():
        """Static access method."""
        if not HandleManager.__instance:
            HandleManager.__instance = HandleManager()
        return HandleManager.__instance

    def __init__(self):
        if self.__instance is not None:
            raise RuntimeError(
                "HandleManager is singleton and " "should be used with getInstance()."
            )
        self.status: Dict[int, EventStatus] = {}  # Handle -> Finished Or Not
        self._last_handle = -1  # should be atomic<int> but it is fine in python
        self.mutex = threading.Lock()
        self.cv = threading.Condition(self.mutex)

    @property
    def last_handle(self):
        return self._last_handle

    def allocate(self) -> int:
        with self.mutex:
            self._last_handle += 1
            self.status[self._last_handle] = EventStatus(EventStatusEnum.ALLOCATED, "")
            return self._last_handle

    def poll(self, handle: int) -> EventStatus:
        with self.mutex:
            return self.status.get(
                handle, EventStatus(EventStatusEnum.UNKNOWN, "Not exist")
            )

    def markDone(self, handle: int):
        with self.mutex:
            self.status[handle] = EventStatus(EventStatusEnum.DONE, "")
            self.cv.notify_all()

    def release(self, handle: int) -> EventStatus:
        with self.mutex:
            return self.status.pop(handle)

    def wait(self, handle: int, timeout: int = None) -> bool:
        def _is_finished():
            return (
                self.status[handle].status == EventStatusEnum.DONE
                or self.status[handle].status == EventStatusEnum.ERROR
            )

        with self.mutex:
            self.cv.wait_for(_is_finished, timeout)

        event_status = self.status[handle]
        if event_status.status == EventStatusEnum.DONE:
            return True

        if event_status.status == EventStatusEnum.ERROR:
            raise RuntimeError(f"Encounter error: {event_status.err}")

        return False

    def _reset(self):
        """This is a danger function and does not handle event notification."""
        with self.mutex:
            self.status = {}
            self._last_handle = -1
