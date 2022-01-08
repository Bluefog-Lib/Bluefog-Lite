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

import dataclasses
import collections
import enum
import selectors
import socket
import struct
import threading
from typing import Any, Deque, Tuple, Optional, Union

from bluefoglite.common.tcp.buffer import Buffer
from bluefoglite.common.tcp.eventloop import EventLoop, Handler
from bluefoglite.common.handle_manager import EventStatus, EventStatusEnum
from bluefoglite.common.logger import logger

# Addresses can be either tuples of varying lengths (AF_INET, AF_INET6,
# AF_NETLINK, AF_TIPC) or strings (AF_UNIX).
TAddress = Union[Tuple[Any, ...], str]

# Socket related constant nomenclature follows:
# AddressFamily startswith 'AF_'
# SocketKind startswith 'SOCK_'
# MsgFlag startswith 'MSG_'
# AddressInfo startswith 'AI_'
# Protocol startswith 'IPPROTO_'
def _get_socket_constants(prefix):
    """Create a dictionary mapping socket module constants to their names."""
    return {getattr(socket, n): n for n in dir(socket) if n.startswith(prefix)}


@dataclasses.dataclass
class SocketFullAddress:
    # For IPv4 address, it is (HOST, PORT).
    addr: TAddress

    # Used for create the socket(family, type, protocol)
    sock_family: int = socket.AF_INET
    sock_type: int = socket.SOCK_STREAM
    sock_protocol: int = socket.IPPROTO_IP


# Fix length-header
Header = collections.namedtuple("Header", ["tag", "content_length"])
# >iI means big-endian Int (4) + unsigned Int (4)
HEADER_FORMAT = ">iI"
HEADER_LENGTH = 8  # must make sure it align with header format


def _create_header(nbytes: int, tag: int = 0) -> bytes:
    """create a message with http style header."""
    assert nbytes > 0
    if nbytes >= 2 ** 32:
        raise ValueError(
            "Don't support to send message length in bytes " f"larger than {2**32}"
        )
    header = Header(tag=tag, content_length=nbytes)
    header_bytes = struct.pack(HEADER_FORMAT, *header)
    return header_bytes


def _phrase_header(head_bytes) -> Header:
    return Header._make(struct.unpack(HEADER_FORMAT, head_bytes))


class PairState(enum.Enum):
    UNKNOWN = 0
    INITIALIZING = 1
    LISTENING = 2
    CONNECTING = 3
    CONNECTED = 4
    CLOSED = 5


class MessageType(enum.Enum):
    SEND_BUFFER = 0
    RECV_BUFFER = 1
    NOTIFY_SEND_READY = 2
    NOTIFY_RECV_READY = 3


@dataclasses.dataclass
class Envelope:
    buf: Buffer
    handle: int
    # Byte offset to read from/write to and byte count.
    offset: int = 0
    nbytes: int = 0


class Pair(Handler):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        event_loop: EventLoop,
        self_rank: int,
        peer_rank: int,
        full_address: SocketFullAddress,
    ):
        self._event_loop = event_loop
        self._state = PairState.INITIALIZING
        self._peer_full_addr: Optional[SocketFullAddress] = None
        self._self_full_addr: SocketFullAddress = full_address
        self.sock: Optional[socket.socket] = None

        # We need mutex since the handle called in the event loop is
        # running in a seperate thread.
        self._mutex = threading.Lock()
        # Used to notify the change of state
        self._cv = threading.Condition(self._mutex)

        self.self_rank = self_rank
        self.peer_rank = peer_rank
        # self.sock: Optional[socket.socket] = None

        # TODO 1. Use the tag to separate the message?
        self._pending_send: Deque[Envelope] = collections.deque()
        self._pending_recv: Deque[Envelope] = collections.deque()

        self.listen()

    @property
    def self_address(self):
        return self._self_full_addr

    @property
    def peer_address(self):
        return self._peer_full_addr

    @property
    def fd(self):
        return self.sock.fileno()

    @property
    def state(self):
        return self._state

    def listen(self):
        """Listen in the port and register handle to selector."""
        with self._mutex:
            self.sock = socket.socket(
                family=self._self_full_addr.sock_family,
                type=self._self_full_addr.sock_type,
                proto=self._self_full_addr.sock_protocol,
            )
            # Set SO_REUSEADDR to allow that reuse of the listening port
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(self._self_full_addr.addr)
            # backlog: queue up as many as 1 connect request
            self.sock.listen(1)

            self.sock.setblocking(False)
            # It is important to overwrite it since the bind addr can choose port 0.
            self._self_full_addr.addr = self.sock.getsockname()

            self.changeState(PairState.LISTENING)
            self._event_loop.register(self.sock, selectors.EVENT_READ, self)

        logger.debug("finished listening register")

    def connect(self, addr: SocketFullAddress):
        """Actively connect to the port.

        Note pair (i, j) both create the listening function in the selector.
        If i < j (i is self-rank and j is peer-rank), do nothing but wait
        If i > j, create a new socket (client) and call connect.
        """
        with self._mutex:
            if self.self_rank == self.peer_rank:
                raise ValueError("Should not connect to self")

            self._peer_full_addr = addr
            if self.self_rank < self.peer_rank:
                # Self is listening side
                # logger.debug(f"{self.self_rank}: waitUntilConnected ")
                self.waitUntilConnected(timeout=None)
                logger.debug("waitUntilConnected done")
            else:
                if self.sock is None:
                    raise RuntimeError("The sock in pair is not created.")
                # Self is connecting side.
                self._event_loop.unregister(self.sock)
                self.sock.close()

                # Recreate a new socket for connecting
                self.sock = socket.socket(
                    family=self._self_full_addr.sock_family,
                    type=self._self_full_addr.sock_type,
                    proto=self._self_full_addr.sock_protocol,
                )
                # Set SO_REUSEADDR to allow that reuse of the listening port
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.changeState(PairState.CONNECTING)

                # Blocked until connected.
                # If we use non-blocking one:
                # connect_ex() is used instead of connect() since connect() would immediately
                # raise a BlockingIOError exception. connect_ex() initially returns an error
                # indicator, errno.EINPROGRESS, instead of raising an exception while the
                # connection is in progress.
                self.sock.setblocking(False)
                self.sock.connect_ex(self._peer_full_addr.addr)

                self.changeState(PairState.CONNECTING)
                self._event_loop.register(
                    self.sock, selectors.EVENT_READ | selectors.EVENT_WRITE, self
                )

                self.waitUntilConnected(timeout=None)

        logger.debug("connect func done")

    def waitUntilConnected(self, timeout=None):
        def pred():
            return self.state == PairState.CONNECTED

        self._cv.wait_for(pred, timeout)

    def waitUntilClosed(self, timeout=None):
        def pred():
            return self.state == PairState.CLOSED

        self._cv.wait_for(pred, timeout)

    def close(self):
        with self._mutex:
            # if self.self_rank < self.peer_rank:
            #     # this is the server side, do nothing but wait the peer close
            #     # then trigger the read event.
            #     logger.error("Close to peer %d as server", self.peer_rank)
            #     self.waitUntilClosed()
            #     logger.error("Close to peer %d as server done", self.peer_rank)
            # else:
            #     logger.error("Close to peer %d as client", self.peer_rank)
            #     # this is the client side, actively close the socket
            try:
                self._event_loop.unregister(self.sock)
                self.sock.close()
            except ValueError as e:
                logger.warning(e)
            finally:
                self.changeState(PairState.CLOSED)
            logger.info("Close to peer %d as client done", self.peer_rank)

    def changeState(self, next_state: PairState):
        if next_state == PairState.CLOSED:
            # Do some cleanup job here
            for envelope in self._pending_recv:
                envelope.buf.handleCompletion(
                    envelope.handle,
                    EventStatus(
                        status=EventStatusEnum.WARN,
                        err="Unfinished send/recv after pair is closed.",
                    ),
                )
            for envelope in self._pending_send:
                envelope.buf.handleCompletion(
                    envelope.handle,
                    EventStatus(
                        status=EventStatusEnum.WARN,
                        err="Unfinished send/recv after pair is closed.",
                    ),
                )

        self._state = next_state
        self._cv.notify_all()

    def handleEvent(self, event: int):
        """Triggered by the selector. Note it is called under a different thread."""
        with self._mutex:
            if self.state == PairState.CLOSED:
                # TODO: 1. properly handle it 2. when closing the pair, cleanup the
                # the events.
                logger.info("Handle the event when the pair is already closed")
                return

            if self.state == PairState.LISTENING:
                # Trigger at the server side when a client socket connect to it
                # The event should be read with no data sent
                self.handleListening(event)
            elif self.state == PairState.CONNECTING:
                # Trigger at the client side when the server accept the connect and
                # reply a ready to read message.
                # The event should be WRITE(send) with no data sent
                self.handleConnecting(event)
            elif self.state == PairState.CONNECTED:
                # For WRITE(send) event, it can be triggered repeatedly.
                # For READ(recv) event, it has two cases:
                #    1. The peer sock actively sent a message OR the message sent
                #       has not be completely received. In this case, sock.recv(xx)
                #       should never return an integer (bytes recieved) equal zero.
                #    2.  The peer socket is closed. It will send an empty read event.
                self.handleConnected(event)
            else:
                logger.warning(
                    "unexpected PairState: %s when handling the event", self.state
                )

    def handleListening(self, event: int):
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        conn, _ = self.sock.accept()
        self._event_loop.unregister(self.sock)

        # Note we change the listening socket to the connected sock
        self.sock = conn
        if not self._peer_full_addr:
            raise RuntimeError(
                "Handle socket listening without proper peer addr setup."
            )
        self._peer_full_addr.addr = self.sock.getpeername()
        self._self_full_addr.addr = self.sock.getsockname()
        self._finishConnected()

    def handleConnecting(self, event: int):
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        # This function is triggered when the remote listening sock accept.
        # we wait util it is connected.
        # So we just check there is no error.
        if self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) != 0:
            logger.error("Get error when try to connect other side of pair.")
            self.close()
            return

        # change from the read | write to read only
        self._event_loop.unregister(self.sock)
        self._finishConnected()

    def handleConnected(self, event: int):
        if event & selectors.EVENT_READ:
            # logger.error("%d, Triggered read", self.self_rank)
            self.read()
        if event & selectors.EVENT_WRITE:
            # logger.error("%d, Triggered write", self.self_rank)
            self.write()

    def _finishConnected(self):
        # Reset the self and peer addr
        self._self_full_addr = self.sock.getsockname()
        self._peer_full_addr = self.sock.getpeername()
        self.sock.setblocking(False)
        # We only need read since we actively send out information
        self._event_loop.register(
            self.sock, selectors.EVENT_READ | selectors.EVENT_WRITE, self
        )

        self.changeState(PairState.CONNECTED)

    def read(self):
        if self._pending_recv:
            # TODO consider supporting rendeveous mode as well as eager mode?
            envolope = self._pending_recv.popleft()
            self._read(envolope)

    def write(self):
        if self._pending_send:
            envolope = self._pending_send.popleft()
            self._write(envolope)

    # _read and _write are long complicated functions due the error handling.
    # See here https://www.python.org/dev/peps/pep-3151/#new-exception-classes
    # for the naming of error that possible thrown by socket in python.

    def _read(self, envelope: Envelope):  # pylint: disable=too-many-branches
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        # TODO: make a queue to send the message seperately
        recv = 0  # number of bytes received
        header = None
        header_bytes = b""
        end_pos = envelope.offset + envelope.nbytes

        logger.debug("handle read envelope %s", envelope)
        while True:
            try:
                # Should be ready to read
                if recv < HEADER_LENGTH:
                    _header = self.sock.recv(HEADER_LENGTH - recv)
                    recv += len(_header)
                    header_bytes += _header

                if header is not None:
                    if envelope.nbytes != -1:
                        start_pos = envelope.offset + recv - HEADER_LENGTH
                        max_recv = min(2048, end_pos - start_pos)
                        num_bytes_recv = self.sock.recv_into(
                            envelope.buf.buffer_view[start_pos:end_pos], max_recv
                        )
                    else:
                        # unspecified buffer.
                        _data = self.sock.recv(2048)
                        num_bytes_recv = len(_data)
                        envelope.buf.data += _data
                else:
                    num_bytes_recv = 0

            except BlockingIOError as e:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                logger.debug("_read encountered %s", e)
                if self.state == PairState.CLOSED:
                    break
            except ConnectionError as e:
                # Other side pair closed the socket.
                logger.warning(
                    "Encountered when recv: %s. Likely, the other "
                    "side of socket closed connection.",
                    e,
                )
                envelope.buf.handleCompletion(
                    envelope.handle,
                    EventStatus(status=EventStatusEnum.ERROR, err=str(e)),
                )
                self.changeState(PairState.CLOSED)

                return
            else:
                recv += num_bytes_recv  # type: ignore

            if len(header_bytes) >= HEADER_LENGTH:
                header = _phrase_header(header_bytes)
                if envelope.nbytes == -1:
                    # Unspecified buffer so no check.
                    pass
                elif header.content_length > envelope.nbytes:
                    raise BufferError(
                        "Recv Buffer size should be equal or "
                        "larger than the sending one."
                    )

            if header is not None:
                content_len = header.content_length
                if recv - HEADER_LENGTH >= content_len:
                    break

        logger.debug("handle read envelope done: %s", envelope)
        envelope.buf.handleCompletion(envelope.handle)

    def _write(self, envelope: Envelope):
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        header = _create_header(envelope.nbytes)
        end_pos = envelope.offset + envelope.nbytes
        sent = 0  # number of bytes sent
        # logger.debug(f"{self.self_rank}: trigged write")

        logger.debug("handle write envelope %s", envelope)
        # TODO: make slot to send the message seperately and concurrently?
        while sent < envelope.nbytes + HEADER_LENGTH:
            try:
                if sent < HEADER_LENGTH:
                    # send header
                    num_bytes_sent = self.sock.send(header[sent:])
                    sent += num_bytes_sent

                if sent >= HEADER_LENGTH:
                    # send content
                    start_pos = envelope.offset + sent - HEADER_LENGTH
                    to_send_bytes = envelope.buf.buffer_view[start_pos:end_pos]
                    num_bytes_sent = self.sock.send(to_send_bytes)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            except ConnectionError as e:
                # Other side pair closed the socket.
                logger.warning(
                    "Encountered when recv: %s. Likely, the other "
                    "side of socket closed connection.",
                    e,
                )
                envelope.buf.handleCompletion(
                    envelope.handle,
                    EventStatus(status=EventStatusEnum.ERROR, err=str(e)),
                )
                self.changeState(PairState.CLOSED)
                return
            else:
                sent += num_bytes_sent
        logger.debug("handle write envelope done: %s", envelope)
        envelope.buf.handleCompletion(envelope.handle)

    def send(  # pylint: disable=too-many-arguments
        self, buf: Buffer, handle: int, nbytes: int, offset: int, slot: int
    ):
        """Send the value in buffer to remote peer in the pair."""
        with self._mutex:
            if self.state != PairState.CONNECTED:
                raise RuntimeError(
                    "The pair socket must be in the CONNECTED state "
                    "before calling the send."
                )

            envelope = Envelope(buf=buf, handle=handle, offset=offset, nbytes=nbytes)
            self._pending_send.append(envelope)

    def recv(  # pylint: disable=too-many-arguments
        self, buf: Buffer, handle: int, nbytes: int, offset: int, slot: int
    ):
        """Send the value in buffer to remote peer in the pair."""
        with self._mutex:
            if self.state != PairState.CONNECTED:
                raise RuntimeError(
                    "The pair socket must be in the CONNECTED state "
                    "before calling the recv."
                )
            envelope = Envelope(buf=buf, handle=handle, offset=offset, nbytes=nbytes)
            self._pending_recv.append(envelope)
