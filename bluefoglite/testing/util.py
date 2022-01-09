import os
import threading
from typing import Callable, List


def multi_thread_help(
    size: int, fn: Callable[[int, int], None], timeout=10
) -> List[Exception]:
    errors: List[Exception] = []

    def wrap_fn(rank, size):
        try:
            os.environ["BFL_WORLD_RANK"] = str(rank)
            os.environ["BFL_WORLD_SIZE"] = str(size)
            fn(rank=rank, size=size)
        except Exception as e:  # pylint: disable=broad-except
            errors.append(e)

    thread_list = [
        threading.Thread(target=wrap_fn, args=(rank, size)) for rank in range(size)
    ]

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join(timeout=timeout)

    return errors
