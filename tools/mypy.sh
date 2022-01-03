#!/usr/bin/env bash

# Usage:  (under the root directory)
#   tools/mypy.sh [--version]

mypy --config-file=tools/.mypy.in "$@" bluefoglite