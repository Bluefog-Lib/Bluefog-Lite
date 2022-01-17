import os
import multiprocessing
import threading
from typing import Callable, List

from bluefoglite.common import const
from bluefoglite.common.logger import Logger


def multi_thread_help(
    size: int, fn: Callable[[int, int], None], timeout=10
) -> List[Exception]:
    errors: List[Exception] = []

    def wrap_fn(rank, size):
        try:
            # Cannot set the env variables since multiple threading shared
            # the same env variables.
            # os.environ[const.BFL_WORLD_RANK] = str(rank)
            # os.environ[const.BFL_WORLD_SIZE] = str(size)
            fn(rank=rank, size=size)
        except Exception as e:  # pylint: disable=broad-except
            errors.append(e)

    thread_list = [
        threading.Thread(target=wrap_fn, args=(rank, size)) for rank in range(size)
    ]

    for t in thread_list:
        t.start()

    for i, t in enumerate(thread_list):
        if i == 0:
            t.join(timeout=timeout)
        else:
            t.join(timeout=1)

    for t in thread_list:
        if t.is_alive():
            errors.append(
                TimeoutError(f"Thread cannot finish within {timeout} seconds.")
            )

    return errors


def multi_process_help(
    size: int, fn: Callable[[int, int], None], timeout=10
) -> List[Exception]:
    errors: List[Exception] = []

    def wrap_fn(rank, size):
        try:
            os.environ[const.BFL_WORLD_RANK] = str(rank)
            os.environ[const.BFL_WORLD_SIZE] = str(size)
            fn(rank=rank, size=size)
        except Exception as e:  # pylint: disable=broad-except
            Logger.get().error(e)

    process_list = [
        multiprocessing.Process(target=wrap_fn, args=(rank, size))
        for rank in range(size)
    ]

    for p in process_list:
        p.daemon = True
        p.start()

    for i, p in enumerate(process_list):
        if i == 0:
            p.join(timeout=timeout)
        else:
            p.join(timeout=1)

    for p in process_list:
        if p.exitcode is not None and p.exitcode != 0:
            errors.append(
                RuntimeError(
                    f"Process didn't finish propoerly -- Exitcode: {p.exitcode}"
                )
            )
            continue
        if p.is_alive():
            errors.append(
                TimeoutError(f"Process cannot finish within {timeout} seconds.")
            )
            p.terminate()

    return errors
