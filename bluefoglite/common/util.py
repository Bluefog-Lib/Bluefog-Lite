# Copyright 2022 Bluefog Team. All Rights Reserved.
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

from typing import Optional
import numpy as np

from bluefoglite.common.tcp import message_pb2


TDtype = int  # int because protobuf encode Enum as integer number


def numpy_to_bfl_dtype(  # pylint: disable=too-many-branches
    np_dtype: np.ScalarType,
) -> TDtype:
    if np_dtype == np.uint8:
        ret_dtype = message_pb2.BFL_UINT8
    elif np_dtype == np.int8:
        ret_dtype = message_pb2.BFL_INT8
    elif np_dtype == np.uint16:
        ret_dtype = message_pb2.BFL_UINT16
    elif np_dtype == np.int16:
        ret_dtype = message_pb2.BFL_INT16
    elif np_dtype == np.int32:
        ret_dtype = message_pb2.BFL_INT32
    elif np_dtype == np.int64:
        ret_dtype = message_pb2.BFL_INT64
    elif np_dtype == np.float16:
        ret_dtype = message_pb2.BFL_FLOAT16
    elif np_dtype == np.float32:
        ret_dtype = message_pb2.BFL_FLOAT32
    elif np_dtype == np.float64:
        ret_dtype = message_pb2.BFL_FLOAT64
    elif np_dtype == np.float128:
        ret_dtype = message_pb2.BFL_FLOAT128
    elif np_dtype == np.bool:
        ret_dtype = message_pb2.BFL_BOOL
    elif np_dtype == np.byte:
        ret_dtype = message_pb2.BFL_BYTE
    else:
        raise ValueError(f"Recieved the unsuported numpy dtype {np_dtype}")
    return ret_dtype


def bfl_to_numpy_dtype(  # pylint: disable=too-many-branches
    bfl_dtype: Optional[TDtype],
) -> np.ScalarType:
    if bfl_dtype == message_pb2.BFL_UINT8:
        ret_dtype = np.uint8
    elif bfl_dtype == message_pb2.BFL_INT8:
        ret_dtype = np.int8
    elif bfl_dtype == message_pb2.BFL_UINT16:
        ret_dtype = np.uint16
    elif bfl_dtype == message_pb2.BFL_INT16:
        ret_dtype = np.int16
    elif bfl_dtype == message_pb2.BFL_INT32:
        ret_dtype = np.int32
    elif bfl_dtype == message_pb2.BFL_INT64:
        ret_dtype = np.int64
    elif bfl_dtype == message_pb2.BFL_FLOAT16:
        ret_dtype = np.float16
    elif bfl_dtype == message_pb2.BFL_FLOAT32:
        ret_dtype = np.float32
    elif bfl_dtype == message_pb2.BFL_FLOAT64:
        ret_dtype = np.float64
    elif bfl_dtype == message_pb2.BFL_FLOAT128:
        ret_dtype = np.float128
    elif bfl_dtype == message_pb2.BFL_BOOL:
        ret_dtype = np.bool
    elif bfl_dtype == message_pb2.BFL_BYTE:
        ret_dtype = np.byte
    else:
        raise ValueError(f"Recieved the unsuported bluefoglit dtype {bfl_dtype}")
    return ret_dtype
