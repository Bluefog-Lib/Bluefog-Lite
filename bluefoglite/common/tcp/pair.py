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
import copy
import enum
import selectors
import socket
import struct
import threading
from typing import Any, Deque, Optional, Tuple, Union

import numpy as np

from bluefoglite.common.tcp import message_pb2  # type: ignore
from bluefoglite.common.tcp.buffer import Buffer, TDtype
from bluefoglite.common.tcp.eventloop import EventLoop, Handler
from bluefoglite.common.handle_manager import (
    BlueFogLiteEventError,
    EventStatus,
    EventStatusEnum,
)
from bluefoglite.common.logger import Logger


MAX_ONE_TIME_RECV_BYTES = 2**20

# ENCODED_HEADER_LENGTH is value determined by proto file, which should be consistent.
ENCODED_HEADER_LENGTH = 11
# It includes the ndim, itemsize, dtype, information, etc. We haven't used that yet!
ENCODED_HEADER_LENGTH_WITH_DETAILS = 32

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

    def copy(self):
        return SocketFullAddress(
            addr=copy.copy(self.addr),
            sock_family=self.sock_family,
            sock_type=self.sock_type,
            sock_protocol=self.sock_protocol,
        )


class PairState(enum.Enum):
    UNKNOWN = 0
    INITIALIZING = 1
    LISTENING = 2
    CONNECTING = 3
    CONNECTED = 4
    CLOSED = 5


@dataclasses.dataclass
class Envelope:  # pylint: disable=too-many-instance-attributes
    message_type: Any  # it is message_pb2.MessageType enum
    buf: Buffer
    handle: int
    # Byte offset to read from/write to and byte count w.r.t local buffer
    # it is not used for remote buffer.
    offset: int
    nbytes: int

    # TODO(ybc) We don't populate the shape because it will make
    # the size of protobuf unknown.
    # Only exists when we use numpy or similar style array
    ndim: Optional[int] = None
    dtype: Optional[TDtype] = None
    itemsize: Optional[int] = None
    num_elements: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None


# Fix length-header
Header = collections.namedtuple("Header", ["tag", "content_length"])
# >iI means big-endian Int (4) + unsigned Int (4)
HEADER_FORMAT = ">iI"
HEADER_LENGTH = 8  # must make sure it align with header format


def _create_header(envelope: Envelope) -> bytes:
    """create a message with http style header."""
    assert envelope.nbytes > 0
    if envelope.nbytes >= 2**32:
        raise ValueError(
            "Don't support to send message length in bytes " f"larger than {2**32}"
        )
    header = Header(tag=0, content_length=envelope.nbytes)
    header_bytes = struct.pack(HEADER_FORMAT, *header)
    return header_bytes


def _phrase_header(head_bytes) -> Header:
    return Header._make(struct.unpack(HEADER_FORMAT, head_bytes))


def _create_pb2_header(envelope: Envelope) -> bytes:
    """create a message with http style header."""
    if envelope.nbytes is None or envelope.nbytes < 0:
        raise ValueError("The nbytpes to send can not be negative.")
    header = message_pb2.Header(
        message_type=envelope.message_type,
        content_length=envelope.nbytes,
        # Other details. Not used yet:
        # ndim=envelope.ndim,
        # dtype=envelope.dtype,
        # itemsize=envelope.itemsize,
        # num_elements=envelope.num_elements,
        #
        # Do not support with shape yet due to varying envelope size:
        # shape = envelope.shape
    )
    return header.SerializeToString()


def _phrase_pb2_header(head_bytes: bytes):
    header = message_pb2.Header()
    header.ParseFromString(head_bytes)
    return header


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
        self._self_full_addr: SocketFullAddress = full_address.copy()
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
    def self_address(self) -> SocketFullAddress:
        return self._self_full_addr

    @property
    def peer_address(self) -> Optional[SocketFullAddress]:
        return self._peer_full_addr

    @property
    def fd(self) -> int:
        if self.sock is None:
            return -1
        return self.sock.fileno()

    @property
    def state(self) -> PairState:
        return self._state

    def listen(self) -> None:
        """Listen in the port and register handle to selector."""
        with self._mutex:
            self.sock = socket.socket(
                family=self._self_full_addr.sock_family,
                type=self._self_full_addr.sock_type,
                proto=self._self_full_addr.sock_protocol,
            )
            # Set SO_REUSEADDR to allow that reuse of the listening port
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            retried = 0
            while retried < 10:
                try:
                    self.sock.bind(self._self_full_addr.addr)
                    break
                except OSError:
                    retried += 1
                    continue
                except Exception as e:
                    Logger.get().error("Failed to bind %s", self._self_full_addr.addr)
                    raise e
            # backlog: queue up as many as 1 connect request
            self.sock.listen(1)

            self.sock.setblocking(False)
            # It is important to overwrite it since the bind addr can choose port 0.
            self._self_full_addr.addr = self.sock.getsockname()
            Logger.get().debug("bind to %s", self._self_full_addr.addr)

            self.changeState(PairState.LISTENING)
            self._event_loop.register(self.sock, selectors.EVENT_READ, self)

    def connect(self, addr: SocketFullAddress) -> None:
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
                self.waitUntilConnected(timeout=None)
                Logger.get().debug("waitUntilConnected done")
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
                Logger.get().debug(
                    "%d call connection to %d, addr %s",
                    self.self_rank,
                    self.peer_rank,
                    self._peer_full_addr.addr,
                )

                self.changeState(PairState.CONNECTING)
                self._event_loop.register(
                    self.sock, selectors.EVENT_READ | selectors.EVENT_WRITE, self
                )

                self.waitUntilConnected(timeout=None)

        Logger.get().debug("connect func done")

    def waitUntilConnected(self, timeout=None) -> None:
        def pred():
            return self.state == PairState.CONNECTED

        self._cv.wait_for(pred, timeout)

    def waitUntilClosed(self, timeout=None) -> None:
        def pred():
            return self.state == PairState.CLOSED

        self._cv.wait_for(pred, timeout)

    def close(self) -> None:
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
            if self.sock is None:
                return
            try:
                self._event_loop.unregister(self.sock)
                self.sock.close()
                self.sock = None
            except ValueError as e:
                Logger.get().warning(e)
            finally:
                self.changeState(PairState.CLOSED)
            Logger.get().info("Close to peer %d as client done", self.peer_rank)

    def changeState(self, next_state: PairState) -> None:
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

    def handleEvent(self, event: int) -> None:
        """Triggered by the selector. Note it is called under a different thread."""
        with self._mutex:
            if self.state == PairState.CLOSED:
                # TODO: 1. properly handle it 2. when closing the pair, cleanup the
                # the events.
                Logger.get().info("Handle the event when the pair is already closed")
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
                Logger.get().warning(
                    "unexpected PairState: %s when handling the event", self.state
                )

    def handleListening(self, event: int) -> None:
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        conn, _ = self.sock.accept()
        self._event_loop.unregister(self.sock)

        # Note we change the listening socket to the connected sock
        self.sock = conn
        if not self._peer_full_addr:
            # The family, protocol, type should be the same.
            self._peer_full_addr = self._self_full_addr
        self._peer_full_addr.addr = self.sock.getpeername()
        self._self_full_addr.addr = self.sock.getsockname()
        self._finishConnected()

    def handleConnecting(self, event: int) -> None:
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        # This function is triggered when the remote listening sock accept.
        # we wait util it is connected.
        # So we just check there is no error.
        _sock_opt = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        if _sock_opt != 0:
            # See the error number meaning in
            # https://www.ibm.com/docs/en/zos/2.2.0?topic=codes-sockets-return-errnos
            Logger.get().error(
                "Get error when try to connect other side of pair: Sockopt: %s.",
                _sock_opt,
            )
            raise BlueFogLiteEventError(
                "Get error when try to connect other side of pair."
            )

        # change from the read | write to read only
        self._event_loop.unregister(self.sock)
        self._finishConnected()

    def handleConnected(self, event: int) -> None:
        """Handles the read/write events when the socket pair is connected.

        Expected the self._mutex is hold during this function is called.
        """
        if event & selectors.EVENT_READ:
            self.read()
        if event & selectors.EVENT_WRITE:
            self.write()

    def _finishConnected(self) -> None:
        # Reset the self and peer addr
        if self.sock is None:
            raise RuntimeError("Unexpected sock is None when finish the connection.")
        self._self_full_addr = self.sock.getsockname()
        self._peer_full_addr = self.sock.getpeername()
        self.sock.setblocking(False)
        # We only need read since we actively send out information
        self._event_loop.register(
            self.sock, selectors.EVENT_READ | selectors.EVENT_WRITE, self
        )

        self.changeState(PairState.CONNECTED)

    def read(self) -> None:
        if self._pending_recv:
            # TODO consider supporting rendeveous mode as well as eager mode?
            envolope = self._pending_recv.popleft()
            self._read(envolope)

    def write(self) -> None:
        if self._pending_send:
            envolope = self._pending_send.popleft()
            self._write(envolope)

    # _read and _write are long complicated functions due the error handling.
    # See here https://www.python.org/dev/peps/pep-3151/#new-exception-classes
    # for the naming of error that possible thrown by socket in python.

    def _read(self, envelope: Envelope) -> None:  # pylint: disable=too-many-branches
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        # TODO: make a queue to send the message seperately
        recv = 0  # number of bytes received
        header = None
        header_bytes = b""
        end_pos = envelope.offset + envelope.nbytes

        Logger.get().debug("handle read envelope %s", envelope)
        # See the comments in the _write function for the exception handling.
        while True:
            try:
                # Should be ready to read
                if recv < ENCODED_HEADER_LENGTH:
                    _header = self.sock.recv(ENCODED_HEADER_LENGTH - recv)
                    recv += len(_header)
                    header_bytes += _header

                if header is not None:
                    if envelope.nbytes != -1:
                        start_pos = envelope.offset + recv - ENCODED_HEADER_LENGTH
                        max_recv = min(MAX_ONE_TIME_RECV_BYTES, end_pos - start_pos)
                        num_bytes_recv = self.sock.recv_into(
                            envelope.buf.buffer_view[start_pos:end_pos], max_recv
                        )
                    else:
                        # unspecified buffer.
                        _data = self.sock.recv(MAX_ONE_TIME_RECV_BYTES)
                        num_bytes_recv = len(_data)
                        envelope.buf.data += _data
                else:
                    num_bytes_recv = 0

            except BlockingIOError as e:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                Logger.get().debug("_read encountered %s", e)
                if self.state == PairState.CLOSED:
                    break
            except BrokenPipeError as e:
                # Other side pair closed the socket.
                Logger.get().warning(
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
            except ConnectionError as e:
                Logger.get().warning(
                    "Encountered when recv: %s. The connection is either refused"
                    "or reset",
                    e,
                )
                raise e
            else:
                recv += num_bytes_recv

            if len(header_bytes) >= ENCODED_HEADER_LENGTH:
                header = _phrase_pb2_header(header_bytes)
                if envelope.nbytes == -1:
                    # Unspecified buffer so no check.
                    pass
                elif (
                    header.content_length > envelope.nbytes  # pylint: disable=no-member
                ):
                    raise BufferError(
                        "Recv Buffer size should be equal or "
                        "larger than the sending one."
                    )

            if header is not None:
                if (
                    recv - ENCODED_HEADER_LENGTH
                    >= header.content_length  # pylint: disable=no-member
                ):
                    break

        Logger.get().debug("handle read envelope done: %s", envelope)
        envelope.buf.handleCompletion(envelope.handle)

    def _write(self, envelope: Envelope) -> None:
        if self.sock is None:
            raise RuntimeError("The sock in pair is not created.")
        # header = _create_header(envelope)
        header = _create_pb2_header(envelope)
        end_pos = envelope.offset + envelope.nbytes
        sent = 0  # number of bytes sent

        Logger.get().debug("handle write envelope %s", envelope)

        # From write(2) man page (NOTES section):
        #
        #  If a write() is interrupted by a signal handler before any
        #  bytes are written, then the call fails with the error EINTR;
        #  if it is interrupted after at least one byte has been written,
        #  the call succeeds, and returns the number of bytes written.
        #
        # Starting from Python 3.3, errors related to socket or address
        # semantics raise OSError or one of its subclasses.
        #    +-- OSError
        #    |    +-- BlockingIOError
        #    |    +-- ChildProcessError
        #    |    +-- ConnectionError
        #    |    |    +-- BrokenPipeError
        #    |    |    +-- ConnectionAbortedError
        #    |    |    +-- ConnectionRefusedError
        #    |    |    +-- ConnectionResetError
        #    |    +-- InterruptedError
        #
        # exception BlockingIOError:
        #    Raised when an operation would block on an object (e.g. socket)
        #    set for non-blocking operation. Corresponds to errno EAGAIN,
        #    EALREADY, EWOULDBLOCK and EINPROGRESS.
        # exception BrokenPipeError
        #    A subclass of ConnectionError, raised when trying to write on a
        #    pipe while the other end has been closed, or trying to write on a
        #    socket which has been shutdown for writing. Corresponds to errno
        #    EPIPE and ESHUTDOWN.
        # exception ConnectionAbortedError
        #    A subclass of ConnectionError, raised when a connection attempt is
        #    aborted by the peer. Corresponds to errno ECONNABORTED.
        # exception ConnectionRefusedError
        #    A subclass of ConnectionError, raised when a connection attempt is
        #    refused by the peer. Corresponds to errno ECONNREFUSED.
        # exception ConnectionResetError
        #    A subclass of ConnectionError, raised when a connection is reset by the
        #    peer. Corresponds to errno ECONNRESET.
        # exception InterruptedError
        #    Raised when a system call is interrupted by an incoming signal.
        #    Corresponds to errno EINTR.
        #    Changed in version 3.5: Python now retries system calls when a syscall
        #    is interrupted by a signal, except if the signal handler raises an exception
        #    (see PEP 475 for the rationale), instead of raising InterruptedError.
        while sent < envelope.nbytes + ENCODED_HEADER_LENGTH:
            try:
                if sent < ENCODED_HEADER_LENGTH:
                    # send header
                    num_bytes_sent = self.sock.send(header[sent:])
                    sent += num_bytes_sent

                if sent >= ENCODED_HEADER_LENGTH:
                    # send content
                    start_pos = envelope.offset + sent - ENCODED_HEADER_LENGTH
                    to_send_bytes = envelope.buf.buffer_view[start_pos:end_pos]
                    num_bytes_sent = self.sock.send(to_send_bytes)
            except BlockingIOError:
                # Resource temporarily unavailable (errno EWOULDBLOCK)
                pass
            except BrokenPipeError as e:
                # Other side pair closed the socket.
                Logger.get().warning(
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
            except ConnectionError as e:
                Logger.get().warning(
                    "Encountered when recv: %s. The connection is either refused"
                    "or reset",
                    e,
                )
                raise e
            else:
                sent += num_bytes_sent
        Logger.get().debug("handle write envelope done: %s", envelope)
        envelope.buf.handleCompletion(envelope.handle)

    def send(  # pylint: disable=too-many-arguments
        self, buf: Buffer, handle: int, nbytes: int, offset: int
    ) -> None:
        """Send the value in buffer to remote peer in the pair."""
        with self._mutex:
            if self.state != PairState.CONNECTED:
                raise RuntimeError(
                    "The pair socket must be in the CONNECTED state "
                    "before calling the send."
                )

            envelope = Envelope(
                message_type=message_pb2.SEND_BUFFER,
                buf=buf,
                handle=handle,
                offset=offset,
                nbytes=nbytes,
                # Numpy-style only
                shape=buf.shape,
                ndim=buf.ndim,
                itemsize=buf.itemsize,
                dtype=buf.dtype,
                num_elements=np.prod(buf.shape),
            )
            self._pending_send.append(envelope)

    def recv(  # pylint: disable=too-many-arguments
        self, buf: Buffer, handle: int, nbytes: int, offset: int
    ) -> None:
        """Send the value in buffer to remote peer in the pair."""
        with self._mutex:
            if self.state != PairState.CONNECTED:
                raise RuntimeError(
                    "The pair socket must be in the CONNECTED state "
                    "before calling the recv."
                )
            envelope = Envelope(
                message_type=message_pb2.RECV_BUFFER,
                buf=buf,
                handle=handle,
                offset=offset,
                nbytes=nbytes,
                # Numpy-style only
                shape=buf.shape,
                ndim=buf.ndim,
                itemsize=buf.itemsize,
                dtype=buf.dtype,
                num_elements=np.prod(buf.shape),
            )
            self._pending_recv.append(envelope)

            # Should we call this immediately since we know it is ready?
            # self.write()
