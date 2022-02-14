from bluefoglite.common.tcp import message_pb2  # type: ignore


def test_serialize():
    h = message_pb2.Header(
        content_length=100,
        tag=0,
        ndim=3,
        shape=[3, 4, 5],
        dtype=message_pb2.BFL_FLOAT64,
    )
    h_bytes = h.SerializeToString()
    h2 = message_pb2.Header()
    h2.ParseFromString(h_bytes)
    assert h == h2
