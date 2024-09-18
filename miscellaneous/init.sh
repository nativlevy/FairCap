virtualenv venv
source ../venv/bin/activate
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_PATH=$(dirname $SCRIPT_DIR)