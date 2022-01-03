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

import os
import socket
import threading
import time
from unittest.mock import MagicMock

from bluefoglite.common.handle_manager import HandleManager
from bluefoglite.common.tcp.buffer import SpecifiedBuffer, UnspecifiedBuffer
from bluefoglite.common.tcp.eventloop import EventLoop
from bluefoglite.common.tcp.pair import Pair, SocketAddress
import numpy as np  # type: ignore
import pytest  # type: ignore


def _multi_thread_help(size, fn, timeout=10):
    errors = []

    def wrap_fn(rank, size):
        try:
            os.environ["BFL_WORLD_RANK"] = str(rank)
            os.environ["BFL_WORLD_SIZE"] = str(size)
            fn(rank=rank, size=size)
        except Exception as e:
            errors.append(e)

    thread_list = [
        threading.Thread(target=wrap_fn, args=(rank, size)) for rank in range(size)
    ]

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join(timeout=timeout)

    return errors


@pytest.fixture
def addr_list():
    return [
        SocketAddress(
            addr=("localhost", 18106 + i),
            sock_family=socket.AF_INET,
            sock_type=socket.SOCK_STREAM,
            sock_protocol=0,
        )
        for i in range(2)
    ]


@pytest.fixture
def array_list():
    n = 9
    return [np.arange(n), np.zeros((n,)).astype(int)]


def test_connect(addr_list):
    def listen_connect_close(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            address=addr_list[rank],
        )
        pair.connect(addr_list[1 - rank])
        pair.close()
        event_loop.close()

    errors = _multi_thread_help(size=2, fn=listen_connect_close)

    for error in errors:
        raise error


def _build_sbuf_from_array(array):
    mock_context = MagicMock()
    return SpecifiedBuffer(mock_context, array.data, array.nbytes)


def test_send_recv_array(addr_list, array_list):
    def send_recv_array(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            address=addr_list[rank],
        )
        pair.connect(addr_list[1 - rank])
        hm = HandleManager.getInstance()

        if rank == 0:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            hm.wait(handle=handle)
        elif rank == 1:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[1])
            pair.recv(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            hm.wait(handle=handle)
            np.testing.assert_allclose(array_list[1], array_list[0])
        pair.close()
        event_loop.close()

    errors = _multi_thread_help(size=2, fn=send_recv_array)

    for error in errors:
        raise error


def test_send_recv_obj(addr_list):
    def send_recv_obj(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            address=addr_list[rank],
        )
        pair.connect(addr_list[1 - rank])
        hm = HandleManager.getInstance()
        data = b"jisdnoldf"

        if rank == 0:
            handle = hm.allocate()
            buf = SpecifiedBuffer(MagicMock(), memoryview(data), len(data))
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            hm.wait(handle=handle)
        elif rank == 1:
            handle = hm.allocate()
            ubuf = UnspecifiedBuffer(MagicMock())
            pair.recv(ubuf, handle, nbytes=-1, offset=0, slot=0)
            hm.wait(handle=handle)
            assert ubuf.data == data
        pair.close()
        event_loop.close()

    errors = _multi_thread_help(size=2, fn=send_recv_obj)

    for error in errors:
        raise error


@pytest.mark.skip("BrokenPipeError: [Errno 32] Broken pipe?")
def test_send_after_close(addr_list, array_list):
    def send_after_close(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            address=addr_list[rank],
        )
        pair.connect(addr_list[1 - rank])
        hm = HandleManager.getInstance()

        if rank == 0:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            time.sleep(0.5)  # wait to send
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            hm.wait(handle=handle)
            pair.close()
        elif rank == 1:
            # Close immediately
            pair.close()

        event_loop.close()

    errors = _multi_thread_help(size=2, fn=send_after_close)

    for error in errors:
        raise error
