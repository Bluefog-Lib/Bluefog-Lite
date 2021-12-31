#!/usr/bin/env bash

# Usage:  (under the root directory)
#   tools/format.sh [--checkonly]

only_print=0

for arg in $@; do
    if [[ "${arg}" == "--checkonly" ]]; then
        only_print=1
    else
        echo -e "\033[31mUnknown arguments. Expected tools/format.sh [--checkonly]\033[0m" >&2
        exit 1
    fi
done

args=()
if (( only_print == 1 )); then
    args+=("--check" "--diff")
fi
format_files=$(find . -name "*.py")

echo "$(black --version)"
LOGS="$(black "${args[@]}" ${format_files} 2>&1)"
STATUS=$?
echo "$LOGS"

exit $STATUS