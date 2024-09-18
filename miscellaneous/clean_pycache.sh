#!usr/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
find $(dirname $SCRIPT_DIR) | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf