from bluefoglite.common.tcp import message_pb2  # type: ignore


def test_serialize():
    h = message_pb2.Header(
        content_length=100,
        message_type=message_pb2.SEND_BUFFER,
        ndim=3,
        shape=[3, 4, 5],
        dtype=message_pb2.BFL_FLOAT64,
        itemsize=8,
        num_elements=3 * 4 * 5,
    )
    h_bytes = h.SerializeToString()
    h2 = message_pb2.Header()
    h2.ParseFromString(h_bytes)
    assert h == h2


def test_none_input():
    h = message_pb2.Header(
        content_length=100,
        message_type=message_pb2.SEND_BUFFER,
        ndim=None,
        shape=None,
        dtype=None,
        itemsize=None,
        num_elements=None,
    )
    h2 = message_pb2.Header(
        content_length=100,
        message_type=message_pb2.SEND_BUFFER,
    )
    # Protobuf will use the default value.
    # pylint: disable=no-member
    assert h.ndim == 0
    assert h.shape == []
    assert h.dtype == 0
    assert h.itemsize == 0
    assert h.num_elements == 0

    assert h == h2
