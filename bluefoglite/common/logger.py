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
from typing import Dict, List, Optional

from bluefoglite.common import const


class DummyLogger:
    def __getattr__(self, name):
        return lambda *x: None


class Logger:
    # In the test (multi-process mode), they shared the same logger
    _should_log: Dict[str, bool] = {}
    _bfl_logger: Optional[logging.Logger] = None
    _dummy_logger = DummyLogger()

    @classmethod
    def get_bfl_logger(cls) -> logging.Logger:
        # We want to initialize it after the Rank, Size are set.
        if cls._bfl_logger:
            return cls._bfl_logger

        bfl_logger = logging.getLogger(const.BFL_LOGGER)

        levels = {
            "critical": logging.CRITICAL,
            "error": logging.ERROR,
            "warn": logging.WARNING,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }
        global_rank = os.getenv(const.BFL_WORLD_RANK)
        set_level = os.getenv(const.BFL_LOG_LEVEL)
        if set_level is None:
            set_level = "warn"
        bfl_logger.setLevel(levels.get(set_level.lower(), '"warn"'))

        ch = logging.StreamHandler()
        ch.setLevel(levels.get(set_level.lower(), '"warn"'))
        formatter = logging.Formatter(
            f"R{global_rank}: %(asctime)-15s %(filename)s:%(lineno)d %(levelname)s  %(message)s"
        )
        ch.setFormatter(formatter)
        bfl_logger.addHandler(ch)
        cls._bfl_logger = bfl_logger
        return cls._bfl_logger

    @classmethod
    def remove_bfl_logger(cls):
        cls._LOGGER_INITIALIZED = False
        cls._bfl_logger = None
        # Is it safe to do so?
        if const.BFL_LOGGER not in logging.Logger.manager.loggerDict:
            return
        del logging.Logger.manager.loggerDict[const.BFL_LOGGER]

    @classmethod
    def checkRanks(cls, ranks: List[str]) -> bool:
        """Check the value for BFL_LOG_RANKS is valid or not."""
        try:
            for rank in ranks:
                world_size = os.getenv(const.BFL_WORLD_SIZE)
                if world_size is None:
                    raise RuntimeError
                if 0 <= int(rank) < int(world_size):
                    continue
                return False
        except RuntimeError as exc:
            raise RuntimeError("BlueFogLite world size is not set.") from exc
        return True

    @classmethod
    def _shouldLogging(cls, log_ranks_str: str) -> bool:
        log_ranks = log_ranks_str.split(",")
        if not cls.checkRanks(log_ranks):
            # The rank is failed to parse, so just always logging
            logging.error(
                "Failed to parse BFL_LOG_RANKS. The format should be "
                "something like `0,1,2` but get %s",
                log_ranks_str,
            )
            return True
        return os.getenv(const.BFL_WORLD_RANK) in log_ranks

    @classmethod
    def shouldLogging(cls) -> bool:
        self_rank_str = os.getenv(const.BFL_WORLD_RANK)
        log_ranks_str = os.getenv(const.BFL_LOG_RANKS)
        if log_ranks_str is None:
            return True
        if self_rank_str is None:
            raise RuntimeError("BlueFogLite rank is not set.")
        if self_rank_str not in cls._should_log:
            cls._should_log[self_rank_str] = cls._shouldLogging(log_ranks_str)
        return cls._should_log[self_rank_str]

    @classmethod
    def get(cls):
        return cls.get_bfl_logger() if cls.shouldLogging() else cls._dummy_logger
