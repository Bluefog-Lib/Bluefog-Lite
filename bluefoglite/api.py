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

from bluefoglite.common.basics import BlueFogLiteGroup

_global_group = BlueFogLiteGroup()

# import basic methods and wrap it with default global group.


def init(group=None):
    if group is None:
        group = _global_group
    group.init()


def shutdown(group=None):
    if group is None:
        group = _global_group
    group.shutdown()


def size(group=None):
    if group is None:
        group = _global_group
    return group.size()


def rank(group=None):
    if group is None:
        group = _global_group
    return group.rank()


def send(dst, obj_or_array, *, tag=0, group=None):
    if group is None:
        group = _global_group
    group.send(dst=dst, obj_or_array=obj_or_array,  tag=tag)


def recv(src, obj_or_array, *, tag=0, group=None):
    if group is None:
        group = _global_group
    return group.recv(src=src, obj_or_array=obj_or_array,  tag=tag)
