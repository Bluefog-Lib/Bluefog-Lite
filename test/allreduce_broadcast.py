import bluefoglite as bfl
import numpy as np  # type: ignore
import pytest  # type: ignore


def setup_module(module):
    bfl.init()


def teardown_module(module):
    # Why do we need to call shutdown explicitly?
    # We need to detect event_loop thread is_alive or not.
    bfl.shutdown()


def test_allreduce_broadcast():
    array = np.array([1.0, 2, 3, 4]) + bfl.rank()
    ret_array = bfl.allreduce(array, agg_op="AVG", inplace=False)
    expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() - 1) / 2
    np.testing.assert_allclose(ret_array, expected_array)

    if bfl.rank() == 0:
        ret_array += 1
    b_ret_array = bfl.broadcast(array=ret_array, root_rank=0, inplace=False)
    expected_array = np.array([1, 2, 3, 4]) + (bfl.size() + 1) / 2
    np.testing.assert_allclose(b_ret_array, expected_array)


def test_allreduce_broadcast_inplace():
    array = np.array([1.0, 2, 3, 4]) + bfl.rank()
    bfl.allreduce(array, agg_op="AVG", inplace=True)
    expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() - 1) / 2
    np.testing.assert_allclose(array, expected_array)

    if bfl.rank() == 0:
        array += 1
    bfl.broadcast(array=array, root_rank=0, inplace=True)
    expected_array = np.array([1, 2, 3, 4]) + (bfl.size() + 1) / 2
    np.testing.assert_allclose(array, expected_array)


def test_allreduce_broadcast_nonblocking():
    array = np.array([1.0, 2, 3, 4]) + bfl.rank()
    fut = bfl.allreduce_nonblocking(array, agg_op="AVG", inplace=False)
    expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() - 1) / 2
    ret_array = fut.result()
    np.testing.assert_allclose(ret_array, expected_array)

    if bfl.rank() == 0:
        ret_array += 1
    fut = bfl.broadcast_nonblocking(array=ret_array, root_rank=0, inplace=False)
    expected_array = np.array([1, 2, 3, 4]) + (bfl.size() + 1) / 2
    np.testing.assert_allclose(fut.result(), expected_array)


def test_allreduce_broadcast_inplace_nonblocking():
    array = np.array([1.0, 2, 3, 4]) + bfl.rank()
    fut = bfl.allreduce_nonblocking(array, agg_op="AVG", inplace=True)
    expected_array = np.array([1.0, 2, 3, 4]) + (bfl.size() - 1) / 2
    fut.result()
    np.testing.assert_allclose(array, expected_array)

    if bfl.rank() == 0:
        array += 1
    fut = bfl.broadcast_nonblocking(array=array, root_rank=0, inplace=True)
    expected_array = np.array([1, 2, 3, 4]) + (bfl.size() + 1) / 2
    fut.result()
    np.testing.assert_allclose(array, expected_array)
