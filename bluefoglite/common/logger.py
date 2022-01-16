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
from typing import List, Optional

from bluefoglite.common import const

_LOGGER_INITIALIZED = False


def list_loggers():
    for nm, lgr in logging.Logger.manager.loggerDict.items():
        print("+ [%-20s] %s " % (nm, lgr))
        if not isinstance(lgr, logging.PlaceHolder):
            for h in lgr.handlers:
                print("     %s" % h)


class logger:
    _should_log: Optional[bool] = None
    _bfl_logger: Optional[logging.Logger] = None

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
        except RuntimeError:
            raise RuntimeError("BlueFogLit world size is not set.")
        return True

    @classmethod
    def _shouldLogging(cls) -> bool:
        log_ranks_str = os.getenv(const.BFL_LOG_RANKS)
        if log_ranks_str is None:
            return True
        ranks = log_ranks_str.split(",")
        print(ranks)
        if not cls.checkRanks(ranks):
            # The rank is failed to parse, so just always logging
            logging.error(
                "Failed to parse BFL_LOG_RANKS. The format should be "
                "something like `0,1,2` but get %s",
                log_ranks_str,
            )
            return True
        return os.getenv(const.BFL_WORLD_RANK) in ranks

    @classmethod
    def shouldLogging(cls) -> bool:
        if cls._should_log is None:
            cls._should_log = cls._shouldLogging()
        return cls._should_log

    @classmethod
    def info(cls, msg: object, *args, **kwargs):
        if cls.shouldLogging():
            bfl_logger = cls.get_bfl_logger()
            bfl_logger.info(msg, *args, **kwargs)

    @classmethod
    def debug(cls, msg: object, *args, **kwargs):
        if cls.shouldLogging():
            bfl_logger = cls.get_bfl_logger()
            bfl_logger.debug(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg: object, *args, **kwargs):
        if cls.shouldLogging():
            bfl_logger = cls.get_bfl_logger()
            bfl_logger.warning(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg: object, *args, **kwargs):
        if cls.shouldLogging():
            bfl_logger = cls.get_bfl_logger()
            bfl_logger.error(msg, *args, **kwargs)

    @classmethod
    def fatal(cls, msg: object, *args, **kwargs):
        if cls.shouldLogging():
            bfl_logger = cls.get_bfl_logger()
            bfl_logger.fatal(msg, *args, **kwargs)
