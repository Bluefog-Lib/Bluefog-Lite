# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: message.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rmessage.proto\x12\x0b\x62luefoglite\"\xb8\x01\n\x06Header\x12\x16\n\x0e\x63ontent_length\x18\x01 \x01(\x06\x12.\n\x0cmessage_type\x18\x02 \x01(\x0e\x32\x18.bluefoglite.MessageType\x12\x0c\n\x04ndim\x18\x03 \x01(\x07\x12!\n\x05\x64type\x18\x04 \x01(\x0e\x32\x12.bluefoglite.DType\x12\x10\n\x08itemsize\x18\x05 \x01(\x07\x12\x14\n\x0cnum_elements\x18\x06 \x01(\x06\x12\r\n\x05shape\x18\x07 \x03(\x05*j\n\x0bMessageType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0f\n\x0bSEND_BUFFER\x10\x01\x12\x0f\n\x0bRECV_BUFFER\x10\x02\x12\x15\n\x11NOTIFY_SEND_READY\x10\x03\x12\x15\n\x11NOTIFY_RECV_READY\x10\x04*\xc2\x01\n\x05\x44Type\x12\r\n\tBFL_UINT8\x10\x00\x12\x0c\n\x08\x42\x46L_INT8\x10\x01\x12\x0e\n\nBFL_UINT16\x10\x02\x12\r\n\tBFL_INT16\x10\x03\x12\r\n\tBFL_INT32\x10\x04\x12\r\n\tBFL_INT64\x10\x05\x12\x0f\n\x0b\x42\x46L_FLOAT16\x10\x06\x12\x0f\n\x0b\x42\x46L_FLOAT32\x10\x07\x12\x0f\n\x0b\x42\x46L_FLOAT64\x10\x08\x12\x10\n\x0c\x42\x46L_FLOAT128\x10\t\x12\x0c\n\x08\x42\x46L_BOOL\x10\n\x12\x0c\n\x08\x42\x46L_BYTE\x10\x0b')

_MESSAGETYPE = DESCRIPTOR.enum_types_by_name['MessageType']
MessageType = enum_type_wrapper.EnumTypeWrapper(_MESSAGETYPE)
_DTYPE = DESCRIPTOR.enum_types_by_name['DType']
DType = enum_type_wrapper.EnumTypeWrapper(_DTYPE)
UNKNOWN = 0
SEND_BUFFER = 1
RECV_BUFFER = 2
NOTIFY_SEND_READY = 3
NOTIFY_RECV_READY = 4
BFL_UINT8 = 0
BFL_INT8 = 1
BFL_UINT16 = 2
BFL_INT16 = 3
BFL_INT32 = 4
BFL_INT64 = 5
BFL_FLOAT16 = 6
BFL_FLOAT32 = 7
BFL_FLOAT64 = 8
BFL_FLOAT128 = 9
BFL_BOOL = 10
BFL_BYTE = 11


_HEADER = DESCRIPTOR.message_types_by_name['Header']
Header = _reflection.GeneratedProtocolMessageType('Header', (_message.Message,), {
  'DESCRIPTOR' : _HEADER,
  '__module__' : 'message_pb2'
  # @@protoc_insertion_point(class_scope:bluefoglite.Header)
  })
_sym_db.RegisterMessage(Header)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MESSAGETYPE._serialized_start=217
  _MESSAGETYPE._serialized_end=323
  _DTYPE._serialized_start=326
  _DTYPE._serialized_end=520
  _HEADER._serialized_start=31
  _HEADER._serialized_end=215
# @@protoc_insertion_point(module_scope)
