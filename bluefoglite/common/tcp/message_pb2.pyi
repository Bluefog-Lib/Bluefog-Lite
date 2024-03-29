"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Generate the derived python file by running the command
  protoc -I=bluefoglite/common/tcp/ --python_out=bluefoglite/common/tcp/ \\
  bluefoglite/common/tcp/message.proto
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _MessageType:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _MessageTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_MessageType.ValueType], builtins.type):  # noqa: F821
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    UNKNOWN: _MessageType.ValueType  # 0
    SEND_BUFFER: _MessageType.ValueType  # 1
    RECV_BUFFER: _MessageType.ValueType  # 2
    NOTIFY_SEND_READY: _MessageType.ValueType  # 3
    NOTIFY_RECV_READY: _MessageType.ValueType  # 4

class MessageType(_MessageType, metaclass=_MessageTypeEnumTypeWrapper): ...

UNKNOWN: MessageType.ValueType  # 0
SEND_BUFFER: MessageType.ValueType  # 1
RECV_BUFFER: MessageType.ValueType  # 2
NOTIFY_SEND_READY: MessageType.ValueType  # 3
NOTIFY_RECV_READY: MessageType.ValueType  # 4
global___MessageType = MessageType

class _DType:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _DTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_DType.ValueType], builtins.type):  # noqa: F821
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    BFL_UINT8: _DType.ValueType  # 0
    BFL_INT8: _DType.ValueType  # 1
    BFL_UINT16: _DType.ValueType  # 2
    BFL_INT16: _DType.ValueType  # 3
    BFL_INT32: _DType.ValueType  # 4
    BFL_INT64: _DType.ValueType  # 5
    BFL_FLOAT16: _DType.ValueType  # 6
    BFL_FLOAT32: _DType.ValueType  # 7
    BFL_FLOAT64: _DType.ValueType  # 8
    BFL_FLOAT128: _DType.ValueType  # 9
    BFL_BOOL: _DType.ValueType  # 10
    BFL_BYTE: _DType.ValueType  # 11

class DType(_DType, metaclass=_DTypeEnumTypeWrapper): ...

BFL_UINT8: DType.ValueType  # 0
BFL_INT8: DType.ValueType  # 1
BFL_UINT16: DType.ValueType  # 2
BFL_INT16: DType.ValueType  # 3
BFL_INT32: DType.ValueType  # 4
BFL_INT64: DType.ValueType  # 5
BFL_FLOAT16: DType.ValueType  # 6
BFL_FLOAT32: DType.ValueType  # 7
BFL_FLOAT64: DType.ValueType  # 8
BFL_FLOAT128: DType.ValueType  # 9
BFL_BOOL: DType.ValueType  # 10
BFL_BYTE: DType.ValueType  # 11
global___DType = DType

class Header(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CONTENT_LENGTH_FIELD_NUMBER: builtins.int
    MESSAGE_TYPE_FIELD_NUMBER: builtins.int
    NDIM_FIELD_NUMBER: builtins.int
    DTYPE_FIELD_NUMBER: builtins.int
    ITEMSIZE_FIELD_NUMBER: builtins.int
    NUM_ELEMENTS_FIELD_NUMBER: builtins.int
    SHAPE_FIELD_NUMBER: builtins.int
    content_length: builtins.int
    """We use the fixed length instead of more efficient Varint
    is to make sure the length of header are fixed.
    See https://developers.google.com/protocol-buffers/docs/encoding.
    """
    message_type: global___MessageType.ValueType
    ndim: builtins.int
    """Extra details"""
    dtype: global___DType.ValueType
    itemsize: builtins.int
    num_elements: builtins.int
    @property
    def shape(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        content_length: builtins.int | None = ...,
        message_type: global___MessageType.ValueType | None = ...,
        ndim: builtins.int | None = ...,
        dtype: global___DType.ValueType | None = ...,
        itemsize: builtins.int | None = ...,
        num_elements: builtins.int | None = ...,
        shape: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["content_length", b"content_length", "dtype", b"dtype", "itemsize", b"itemsize", "message_type", b"message_type", "ndim", b"ndim", "num_elements", b"num_elements"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["content_length", b"content_length", "dtype", b"dtype", "itemsize", b"itemsize", "message_type", b"message_type", "ndim", b"ndim", "num_elements", b"num_elements", "shape", b"shape"]) -> None: ...

global___Header = Header
