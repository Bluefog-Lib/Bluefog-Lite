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

from bluefoglite.common import const

global_rank = os.getenv(const.BFL_WORLD_RANK)
logger = logging.getLogger(const.BFL_LOGGER)
levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
set_level = os.getenv(const.BFL_LOG_LEVEL)
if set_level is None:
    set_level = "warn"
logger.setLevel(levels.get(set_level.lower(), '"warn"'))

ch = logging.StreamHandler()
ch.setLevel(levels.get(set_level.lower(), '"warn"'))
formatter = logging.Formatter(
    f"R{global_rank}: %(asctime)-15s %(filename)s:%(lineno)d %(levelname)s  %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
