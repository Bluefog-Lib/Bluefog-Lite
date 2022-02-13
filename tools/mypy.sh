#!/usr/bin/env bash

# Usage:  (under the root directory)
#   tools/mypy.sh [--version]

mypy_files=$(find bluefoglite -name "*.py" ! -name '*_pb2.py')
mypy --config-file=tools/.mypy.in "$@" ${mypy_files}
