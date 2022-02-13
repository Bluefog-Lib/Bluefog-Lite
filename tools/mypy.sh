#!/usr/bin/env bash

# Usage:  (under the root directory)
#   tools/mypy.sh [--version]

# mypy_files=$(find bluefoglite -name "*.py" ! -name '*_pb2.py')
# mypy --config-file=tools/.mypy.in "$@" ${mypy_files}
mypy --config-file=tools/.mypy.in bluefoglite/*.py bluefoglite/common/*.py bluefoglite/common/collective_comm/*.py