import itertools
from typing import Dict


import numpy as np

from bluefoglite.common.tcp.buffer import Buffer, NumpyBuffer


def neighbor_allreduce(  # pylint: disable=too-many-locals
    in_buf: NumpyBuffer,
    out_buf: NumpyBuffer,
    self_weight: float,
    src_weights: Dict[int, float],
    dst_weights: Dict[int, float],
) -> None:
    if src_weights is None or dst_weights is None:
        raise ValueError("must provide src_weights or dst_weights")

    # assert in_buf.buffer_length * (len(src_weights) + 1) == out_buf.buffer_length
    # Extra buffer for receiving the information from neighbors
    tmp_shape = list(int(s) for s in in_buf.shape)
    tmp_shape[0] *= len(src_weights)
    tmp_buf = in_buf.create_new_buffer(shape=tuple(tmp_shape))

    send_handle = []
    extra_send_bufs = []
    for rank, weight in dst_weights.items():
        if weight != 1:
            extra_send_buf = in_buf.clone()
            extra_send_buf.mul_(weight)
            extra_send_bufs.append(extra_send_buf)
        else:
            extra_send_buf = in_buf
        handle = extra_send_buf.isend(rank)
        send_handle.append(handle)

    nbytes = in_buf.buffer_length
    recv_handle = []
    for i, rank in enumerate(sorted(src_weights.keys())):
        handle = tmp_buf.irecv(rank, nbytes=nbytes, offset=i * nbytes)
        recv_handle.append(handle)

    for h in itertools.chain(send_handle, recv_handle):
        Buffer.waitCompletion(h)

    # Be careful that we want to overwrite the memory that buffer_view pointed to
    # not to change the object buffer_view reference. [:] is necessary here.
    out_buf.buffer_view[:] = (in_buf.array * self_weight).data.cast("c")
    self_array = np.frombuffer(out_buf.buffer_view[:nbytes], dtype=out_buf.dtype)
    for i, rank in enumerate(sorted(src_weights.keys())):
        weight = src_weights[rank]
        offset = i * nbytes
        neighbor_tmp_array = np.frombuffer(
            tmp_buf.buffer_view[offset : offset + nbytes], dtype=out_buf.dtype
        )
        self_array += neighbor_tmp_array * weight

    del tmp_buf
    del extra_send_bufs
