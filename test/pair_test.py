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

import logging
import os
import socket
import threading
import time
from unittest.mock import MagicMock

import numpy as np  # type: ignore
import pytest  # type: ignore

from bluefoglite import BlueFogLiteEventError

from bluefoglite.common.handle_manager import HandleManager
from bluefoglite.common.tcp.buffer import SpecifiedBuffer, UnspecifiedBuffer
from bluefoglite.common.tcp.eventloop import EventLoop
from bluefoglite.common.tcp.pair import Pair, SocketFullAddress
from bluefoglite.testing.util import multi_thread_help


@pytest.fixture(name="full_addr_list")
def fixture_full_addr_list(num=2):
    base_port = 18106
    while base_port < 2 ** 16:
        full_address_list = [
            SocketFullAddress(
                addr=("localhost", base_port + i),
                sock_family=socket.AF_INET,
                sock_type=socket.SOCK_STREAM,
                sock_protocol=0,
            )
            for i in range(num)
        ]
        # Make sure the above ports are available. If not
        # try a new base port.
        try:
            for address in full_address_list:
                sock = socket.socket(address.sock_family, address.sock_type)
                sock.bind(address.addr)
                sock.close()
        except OSError:
            base_port += num
            continue
        break
    return full_address_list


@pytest.fixture(name="array_list")
def fixture_array_list():
    n = 9
    return [np.arange(n), np.zeros((n,)).astype(int)]


def test_connect(full_addr_list):
    def listen_connect_close(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=full_addr_list[rank],
        )
        pair.connect(full_addr_list[1 - rank])
        pair.close()
        event_loop.close()

    errors = multi_thread_help(size=2, fn=listen_connect_close)

    for error in errors:
        raise error


def _build_sbuf_from_array(array):
    mock_context = MagicMock()
    return SpecifiedBuffer(mock_context, array.data, array.nbytes)


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_send_recv_array(full_addr_list, array_list, reverse_send_recv):
    def send_recv_array(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=full_addr_list[rank],
        )
        pair.connect(full_addr_list[1 - rank])
        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        if rank == send_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            hm.wait(handle=handle)
        elif rank == recv_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[1])
            pair.recv(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            hm.wait(handle=handle)
            np.testing.assert_allclose(array_list[1], array_list[0])
        pair.close()
        event_loop.close()

    errors = multi_thread_help(size=2, fn=send_recv_array)

    for error in errors:
        raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_send_recv_obj(full_addr_list, reverse_send_recv):
    def send_recv_obj(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=full_addr_list[rank],
        )
        pair.connect(full_addr_list[1 - rank])
        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

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

    errors = multi_thread_help(size=2, fn=send_recv_obj)

    for error in errors:
        raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_send_after_peer_close(full_addr_list, array_list, reverse_send_recv):
    def send_after_peer_close(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=full_addr_list[rank],
        )
        pair.connect(full_addr_list[1 - rank])
        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        if rank == send_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            time.sleep(0.5)  # wait to send
            try:
                pair.send(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
                hm.wait(handle=handle)
            except BlueFogLiteEventError:
                # Encounter error: [Errno 32] Broken pipe
                pass
            pair.close()
        elif rank == recv_rank:
            # Close immediately
            pair.close()

        event_loop.close()

    errors = multi_thread_help(size=2, fn=send_after_peer_close)

    for error in errors:
        raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_close_before_send_finish(
    full_addr_list, array_list, reverse_send_recv, caplog
):
    def close_before_send_finish(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=full_addr_list[rank],
        )
        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        pair.connect(full_addr_list[1 - rank])
        # Close immediate after Send
        if rank == send_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.send(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            pair.close()
            # Close it withour wait sending.
            with caplog.at_level(logging.WARNING):
                hm.wait(handle=handle)

            assert len(caplog.record_tuples) == 1
            assert caplog.record_tuples[0][:2] == ("BFL_LOGGER", logging.WARNING)
            assert (
                "Unfinished send/recv after pair is closed"
                in caplog.record_tuples[0][2]
            )
        elif rank == recv_rank:
            time.sleep(0.1)
            pair.close()

        event_loop.close()

    errors = multi_thread_help(size=2, fn=close_before_send_finish)

    for error in errors:
        raise error


@pytest.mark.parametrize("reverse_send_recv", [True, False])
def test_close_before_recv_finish(
    full_addr_list, array_list, reverse_send_recv, caplog
):
    def close_before_recv_finish(rank, size):
        event_loop = EventLoop()
        event_loop.run()
        pair = Pair(
            event_loop=event_loop,
            self_rank=rank,
            peer_rank=1 - rank,
            full_address=full_addr_list[rank],
        )
        hm = HandleManager.getInstance()
        send_rank, recv_rank = (0, 1) if reverse_send_recv else (1, 0)

        pair.connect(full_addr_list[1 - rank])
        if rank == send_rank:
            time.sleep(0.1)
            pair.close()
        elif rank == recv_rank:
            handle = hm.allocate()
            buf = _build_sbuf_from_array(array_list[0])
            pair.recv(buf, handle, nbytes=buf.buffer_length, offset=0, slot=0)
            pair.close()
            # Close it without wait receiving.
            with caplog.at_level(logging.WARNING):
                hm.wait(handle=handle)

            assert len(caplog.record_tuples) == 1
            assert caplog.record_tuples[0][:2] == ("BFL_LOGGER", logging.WARNING)
            assert (
                "Unfinished send/recv after pair is closed"
                in caplog.record_tuples[0][2]
            )

        event_loop.close()

    errors = multi_thread_help(size=2, fn=close_before_recv_finish)

    for error in errors:
        raise error
