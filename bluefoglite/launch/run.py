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

import argparse
import atexit
from datetime import datetime
import glob
import os
import signal
import subprocess
import traceback
import sys

import bluefoglite as bfl


def parse_args():
    parser = argparse.ArgumentParser(description="BluefogLite Launcher")

    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        dest="version",
        help="Shows bluefog version.",
    )

    parser.add_argument(
        "-np",
        "--num-proc",
        action="store",
        dest="np",
        type=int,
        help="Total number of training processes.",
    )

    parser.add_argument(
        "--master-port",
        action="store",
        dest="master_port",
        type=int,
        default=29500,
        help="Master port for BluefogLite.",
    )

    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="Command to be executed."
    )

    parsed_args = parser.parse_args()

    if not parsed_args.version and not parsed_args.np:
        parser.error("argument -np/--num-proc is required")

    return parsed_args


def _maybe_kill_process(pid_list):
    for pid in pid_list:
        try:
            os.kill(pid, signal.SIGINT)
        except OSError:
            # PID doesn't exist
            pass


def cleanup(shared_file_dir):
    if os.path.exists(shared_file_dir):
        for f in glob.glob(os.path.join(shared_file_dir, "*")):
            try:
                os.remove(f)
            except OSError as e:
                # TODO logs to warning
                del e
        os.rmdir(shared_file_dir)


def main():
    args = parse_args()

    if args.version:
        print(bfl.__version__)
        sys.exit(0)

    p_ctx_list, pid_list = [], []
    runtime_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    shared_file_dir = os.path.join("/tmp", ".bluefoglite", runtime_str)
    if not os.path.exists(shared_file_dir):
        os.makedirs(shared_file_dir)

    atexit.register(cleanup, shared_file_dir)

    for i in range(args.np):
        env = os.environ.copy()
        env["BFL_WORLD_RANK"] = str(i)
        env["BFL_WORLD_SIZE"] = str(args.np)
        env["BFL_FILE_STORE"] = shared_file_dir
        # TODO fix this
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(args.master_port)

        stdout = None
        stderr = subprocess.STDOUT
        p_ctx = subprocess.Popen(  # pylint: disable=consider-using-with
            args.command, shell=False, env=env, stdout=stdout, stderr=stderr
        )
        p_ctx_list.append(p_ctx)
        pid_list.append(p_ctx.pid)

    def handler(signum, frame):
        _maybe_kill_process(pid_list)
        cleanup(shared_file_dir)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    atexit.register(_maybe_kill_process, pid_list)
    timeout = 2

    while any(p_ctx.poll() is None for p_ctx in p_ctx_list):
        try:
            [p_ctx.wait(timeout=timeout) for p_ctx in p_ctx_list]
        except subprocess.TimeoutExpired:
            pass
        except:  # pylint: disable=bare-except
            traceback.print_exc()
            _maybe_kill_process(pid_list)
            break

    for i, p_ctx in enumerate(p_ctx_list):
        if p_ctx.poll() != 0:
            print(f"Rank {i} stopped with nonzero returncode.")


if __name__ == "__main__":
    main()
