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

echo ""
echo ""
echo "===================Running pytest_profile==================="
tools/profile_test.sh
