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

import copy
import glob
import os
import pickle
import threading
import time
from typing import Any, List, Optional
from bluefoglite.common.logger import Logger


class FileStore:
    def __init__(self, path: str):
        self.path = path
        # To avoid mis-write
        assert "bluefoglite" in self.path

    def _get_key_path(self, key):
        assert isinstance(key, str)
        return os.path.join(self.path, key)

    def set(self, key: str, value: Any):
        if self._check_exist(key):
            Logger.get().warning("FileStore: overwrite the file")
        with open(self._get_key_path(key), "wb") as f:
            pickle.dump(value, f)

    def _check_exist(self, key: str) -> bool:
        return os.path.exists(self._get_key_path(key))

    def get(self, key: str, timeout: float = 100.0):
        self.wait([key], timeout)
        with open(self._get_key_path(key), "rb") as f:
            tried = 0
            while True:
                try:
                    value = pickle.load(f)
                    return value
                except EOFError as e:
                    # Possible the other process just recreate the file but
                    # not finished the writting yet.
                    time.sleep(0.01)
                    tried += 1
                    if tried > 10:
                        raise e

    def delete(self, key: str) -> bool:
        if not self._check_exist(key):
            return False
        os.remove(self._get_key_path(key))
        return True

    def wait(self, keys: List[str], timeout: Optional[float] = None):
        timeout_t = time.time() + timeout if timeout is not None else None
        remain_keys = [key for key in keys if not self._check_exist(key)]
        while remain_keys:
            if timeout_t is not None and time.time() > timeout_t:
                raise KeyError(f"Timeout for waiting the keys {remain_keys}")
            time.sleep(0.05)
            remain_keys = [key for key in keys if not self._check_exist(key)]

    def close(self):
        if os.path.exists(self.path):
            self.reset()
            os.rmdir(self.path)

    def reset(self):
        for f in glob.glob(os.path.join(self.path, "*")):
            try:
                os.remove(f)
            except OSError as e:
                Logger.get().debug("When reset the file store encountered: %s", e)


class InMemoryStore:
    def __init__(self):
        self._mutex = threading.Lock()
        self._cv = threading.Condition(self._mutex)
        self.store = {}

    def get(self, key: str, timeout: float = 100.0):
        self.wait([key], timeout)
        with self._mutex:
            return self.store.get(key, None)

    def set(self, key: str, value: Any):
        with self._mutex:
            self.store[key] = copy.copy(value)
            self._cv.notify_all()

    def delete(self, key: str) -> bool:
        with self._mutex:
            if key not in self.store:
                return False
            self.store.pop(key)
            self._cv.notify_all()
            return True

    def wait(self, keys: List[str], timeout: Optional[float] = None):
        timeout_t = time.time() + timeout if timeout is not None else None
        remain_keys = [key for key in keys if key not in self.store]
        with self._mutex:
            while remain_keys:
                wait_timeout = (
                    timeout_t - time.time() if timeout_t is not None else None
                )
                if wait_timeout and wait_timeout < 0:
                    break
                self._cv.wait(timeout=wait_timeout)

                remain_keys = [key for key in keys if key not in self.store]

        if remain_keys:
            raise KeyError(f"Timeout for waiting the keys {remain_keys}")

    def close(self):
        self.reset()

    def reset(self):
        with self._mutex:
            self.store.clear()
            self._cv.notify_all()
