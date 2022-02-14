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

import itertools
import threading
import time

import pytest  # type: ignore

from bluefoglite.common.handle_manager import HandleManager


@pytest.mark.parametrize("num_thread,incr", itertools.product([2, 4, 6], [3, 40, 5]))
def test_handle_manager_allocate(num_thread, incr):
    hm = HandleManager.getInstance()
    hm._reset()  # pylint: disable=protected-access

    def allocate(incr):
        prev_handle = -1
        hm = HandleManager.getInstance()
        for _ in range(incr):
            handle = hm.allocate()
            assert handle > prev_handle
            # print(handle)
            prev_handle = handle
            time.sleep(0.01)

    thread_list = [
        threading.Thread(target=allocate, args=(incr,)) for _ in range(num_thread)
    ]

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    assert hm.last_handle == num_thread * incr - 1
    hm._reset()  # pylint: disable=protected-access


@pytest.mark.parametrize("sleep_time", [0.5, 1, 1.5])
def test_handle_manager_wait(sleep_time):
    hm = HandleManager.getInstance()
    hm._reset()  # pylint: disable=protected-access

    def allocate_markdone():
        hm = HandleManager.getInstance()

        handle = hm.allocate()
        time.sleep(sleep_time)
        hm.markDone(handle)

    t = threading.Thread(target=allocate_markdone)
    t.start()

    hm = HandleManager.getInstance()
    t_start = time.time()
    hm.wait(handle=hm.last_handle)
    wait_time = time.time() - t_start

    t.join()

    assert wait_time >= sleep_time - 0.1
    assert wait_time <= sleep_time + 0.2


def test_handle_manager_wait_timeout():
    sleep_time = 1.0
    timeout_time = 0.5
    hm = HandleManager.getInstance()
    hm._reset()  # pylint: disable=protected-access

    def allocate_markdone():
        hm = HandleManager.getInstance()

        handle = hm.allocate()
        time.sleep(sleep_time)
        hm.markDone(handle)

    t = threading.Thread(target=allocate_markdone)
    t.start()

    hm = HandleManager.getInstance()
    assert not hm.wait(handle=hm.last_handle, timeout=timeout_time)

    t.join()
