import itertools
import threading
import time

from bluefoglite.common.handle_manager import HandleManager
import pytest  # type: ignore


@pytest.mark.parametrize(
    "num_thread,incr",
    itertools.product([2, 4, 6], [3, 40, 5]),
)
def test_handle_manager_allocate(num_thread, incr):
    hm = HandleManager.getInstance()

    def allocate(incr):
        prev_handle = -1
        hm = HandleManager.getInstance()
        for _ in range(incr):
            handle = hm.allocate()
            assert handle > prev_handle
            # print(handle)
            prev_handle = handle
            time.sleep(0.01)

    thread_list = [threading.Thread(target=allocate, args=(incr,))
                   for _ in range(num_thread)]

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    assert hm.last_handle == num_thread * incr - 1
    hm._reset()


@pytest.mark.parametrize("sleep_time", [0.5, 1, 1.5])
def test_handle_manager_wait(sleep_time):

    def allocate_markdone():
        hm = HandleManager.getInstance()

        handle = hm.allocate()
        time.sleep(sleep_time)
        hm.markDone(handle)

    t = threading.Thread(target=allocate_markdone)
    t.start()

    hm = HandleManager.getInstance()
    t_start = time.time()
    hm.wait(handle=hm.last_handle)
    wait_time = time.time() - t_start

    t.join()

    assert wait_time >= sleep_time - 0.1
    assert wait_time <= sleep_time + 0.2


def test_handle_manager_wait_timeout():
    sleep_time = 1.0
    timeout_time = 0.5

    def allocate_markdone():
        hm = HandleManager.getInstance()

        handle = hm.allocate()
        time.sleep(sleep_time)
        hm.markDone(handle)

    t = threading.Thread(target=allocate_markdone)
    t.start()

    hm = HandleManager.getInstance()
    assert hm.wait(handle=hm.last_handle, timeout=timeout_time) == False

    t.join()
