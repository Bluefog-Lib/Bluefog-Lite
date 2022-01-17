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
import pytest  # type: ignore

from bluefoglite.common import const
from bluefoglite.common.logger import Logger


@pytest.fixture(name="setup")
def fixture_setup():
    # Nothing to setup but we need teardown.
    yield None
    # Test purpose only -- reset the logger.
    Logger.remove_bfl_logger()
    # Restore the os.environment
    os.environ.pop(const.BFL_LOG_LEVEL, None)
    os.environ.pop(const.BFL_WORLD_SIZE, None)
    os.environ.pop(const.BFL_LOG_RANKS, None)


def test_normal_logging(setup, caplog):
    os.environ[const.BFL_LOG_LEVEL] = "debug"
    os.environ[const.BFL_WORLD_RANK] = "0"

    with caplog.at_level(logging.DEBUG):
        Logger.get().debug("Test 1")
        Logger.get().info("Test 2")
        Logger.get().warning("Test 3")
        Logger.get().error("Test 4")
        Logger.get().fatal("Test 5")

    assert len(caplog.record_tuples) == 5
    assert caplog.record_tuples[0] == ("BFL_LOGGER", logging.DEBUG, "Test 1")
    assert caplog.record_tuples[1] == ("BFL_LOGGER", logging.INFO, "Test 2")
    assert caplog.record_tuples[2] == ("BFL_LOGGER", logging.WARNING, "Test 3")
    assert caplog.record_tuples[3] == ("BFL_LOGGER", logging.ERROR, "Test 4")
    assert caplog.record_tuples[4] == ("BFL_LOGGER", logging.FATAL, "Test 5")


def test_logging_level(setup, caplog):
    os.environ[const.BFL_LOG_LEVEL] = "error"
    os.environ[const.BFL_WORLD_RANK] = "0"

    with caplog.at_level(logging.DEBUG):
        Logger.get().debug("Test 1")
        Logger.get().info("Test 2")
        Logger.get().warning("Test 3")
        Logger.get().error("Test 4")
        Logger.get().fatal("Test 5")

    assert len(caplog.record_tuples) == 2
    assert caplog.record_tuples[0] == ("BFL_LOGGER", logging.ERROR, "Test 4")
    assert caplog.record_tuples[1] == ("BFL_LOGGER", logging.FATAL, "Test 5")


def test_log_ranks(setup, caplog):
    os.environ[const.BFL_LOG_LEVEL] = "debug"
    os.environ[const.BFL_WORLD_SIZE] = "4"
    os.environ[const.BFL_LOG_RANKS] = "1,2"

    with caplog.at_level(logging.DEBUG):
        os.environ[const.BFL_WORLD_RANK] = "0"
        Logger.get().debug("Test 1")  # Not logged

        os.environ[const.BFL_WORLD_RANK] = "1"
        Logger.get().debug("Test 2")  # logged

        os.environ[const.BFL_WORLD_RANK] = "2"
        Logger.get().debug("Test 3")  # logged

        os.environ[const.BFL_WORLD_RANK] = "3"
        Logger.get().debug("Test 4")  # not logged

    assert len(caplog.record_tuples) == 2
    assert caplog.record_tuples[0] == ("BFL_LOGGER", logging.DEBUG, "Test 2")
    assert caplog.record_tuples[1] == ("BFL_LOGGER", logging.DEBUG, "Test 3")
