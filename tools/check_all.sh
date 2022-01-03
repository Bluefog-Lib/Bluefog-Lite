#!/usr/bin/env bash
echo "===================Running format================="
tools/format.sh --checkonly

echo ""
echo ""
echo "===================Running lint==================="
tools/lint.sh

echo ""
echo ""
echo "===================Running mypy==================="
tools/mypy.sh
