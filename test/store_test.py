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

import atexit
import datetime
import os
import time
import threading

from bluefoglite.common.store import InMemoryStore, FileStore

import pytest  # type: ignore

runtime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
shared_file_dir = os.path.join('/tmp', '.bluefoglite', runtime_str)
if not os.path.exists(shared_file_dir):
    os.makedirs(shared_file_dir)
STORES = [InMemoryStore(), FileStore(shared_file_dir)]


def cleanup():
    for store in STORES:
        store.close()


atexit.register(cleanup)


def _multi_thread_help(size, fn, timeout=10):
    errors = []

    def wrap_fn(rank, size):
        try:
            os.environ['BFL_WORLD_RANK'] = str(rank)
            os.environ['BFL_WORLD_SIZE'] = str(size)
            fn(rank=rank, size=size)
        except Exception as e:
            errors.append(e)

    thread_list = [threading.Thread(target=wrap_fn, args=(rank, size))
                   for rank in range(size)]

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join(timeout=timeout)

    return errors


@pytest.mark.parametrize("store", STORES)
def test_set_get(store):
    key = "fake_key"
    value = "Fake Value"

    def fn(rank, size):
        try:
            if rank == 0:
                time.sleep(0.1)
                store.set(key, value)
            elif rank == 1:
                get_value = store.get(key)
                assert get_value == value
            else:
                pass
        except Exception as e:
            errors.append(e)

    errors = _multi_thread_help(size=2, fn=fn, timeout=2)

    store.reset()

    for error in errors:
        raise error


@pytest.mark.parametrize("store", STORES)
def test_set_get_multiple(store):
    key = "key_"
    value = "value_"

    def fn(rank, size):
        try:
            store.set(key+str(rank), value+str(rank))
            for i in range(size):
                get_value = store.get(key+str(i))
                assert get_value == value+str(i)
        except Exception as e:
            errors.append(e)

    errors = _multi_thread_help(size=4, fn=fn, timeout=2)

    store.reset()

    for error in errors:
        raise error


@pytest.mark.parametrize("store", STORES)
def test_get_timeout(store):
    with pytest.raises(KeyError):
        store.get(key="no_exist", timeout=1)
