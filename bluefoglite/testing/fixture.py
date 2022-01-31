import datetime
import os

import pytest  # type: ignore

from bluefoglite.common.store import FileStore


@pytest.fixture(name="store")
def fixture_store():
    runtime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    shared_file_dir = os.path.join("/tmp", ".bluefoglite", __name__, runtime_str)
    if not os.path.exists(shared_file_dir):
        os.makedirs(shared_file_dir)
    f_store = FileStore(shared_file_dir)
    yield f_store
    f_store.close()
