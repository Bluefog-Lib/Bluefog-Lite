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
import logging
import os

from bluefoglite.common import const
from bluefoglite.common.logger import logger


def test_normal_logging(caplog):
    os.environ[const.BFL_LOG_LEVEL] = "debug"
    os.environ[const.BFL_WORLD_RANK] = "0"

    with caplog.at_level(logging.DEBUG):
        logger.debug("Test 1")
        logger.info("Test 2")
        logger.warning("Test 3")
        logger.error("Test 4")
        logger.fatal("Test 5")

    assert len(caplog.record_tuples) == 5
    assert caplog.record_tuples[0] == ("BFL_LOGGER", logging.DEBUG, "Test 1")
    assert caplog.record_tuples[1] == ("BFL_LOGGER", logging.INFO, "Test 2")
    assert caplog.record_tuples[2] == ("BFL_LOGGER", logging.WARNING, "Test 3")
    assert caplog.record_tuples[3] == ("BFL_LOGGER", logging.ERROR, "Test 4")
    assert caplog.record_tuples[4] == ("BFL_LOGGER", logging.FATAL, "Test 5")

    logger.remove_bfl_logger()  # Test purpose only -- reset the logger.


def test_logging_level(caplog):
    os.environ[const.BFL_LOG_LEVEL] = "error"
    os.environ[const.BFL_WORLD_RANK] = "0"

    with caplog.at_level(logging.DEBUG):
        logger.debug("Test 1")
        logger.info("Test 2")
        logger.warning("Test 3")
        logger.error("Test 4")
        logger.fatal("Test 5")

    assert len(caplog.record_tuples) == 2
    assert caplog.record_tuples[0] == ("BFL_LOGGER", logging.ERROR, "Test 4")
    assert caplog.record_tuples[1] == ("BFL_LOGGER", logging.FATAL, "Test 5")

    logger.remove_bfl_logger()  # Test purpose only -- reset the logger.


def test_log_ranks(caplog):
    os.environ[const.BFL_LOG_LEVEL] = "debug"
    os.environ[const.BFL_WORLD_SIZE] = "4"
    os.environ[const.BFL_LOG_RANKS] = "1,2"

    with caplog.at_level(logging.DEBUG):
        logger._should_log = None  # pylint: disable=protected-access
        os.environ[const.BFL_WORLD_RANK] = "0"
        logger.debug("Test 1")  # Not logged

        logger._should_log = None  # pylint: disable=protected-access
        os.environ[const.BFL_WORLD_RANK] = "1"
        logger.debug("Test 2")  # logged

        logger._should_log = None  # pylint: disable=protected-access
        os.environ[const.BFL_WORLD_RANK] = "2"
        logger.debug("Test 3")  # logged

        logger._should_log = None  # pylint: disable=protected-access
        os.environ[const.BFL_WORLD_RANK] = "3"
        logger.debug("Test 4")  # not logged

    assert len(caplog.record_tuples) == 2
    assert caplog.record_tuples[0] == ("BFL_LOGGER", logging.DEBUG, "Test 2")
    assert caplog.record_tuples[1] == ("BFL_LOGGER", logging.DEBUG, "Test 3")

    logger.remove_bfl_logger()  # Test purpose only -- reset the logger.
