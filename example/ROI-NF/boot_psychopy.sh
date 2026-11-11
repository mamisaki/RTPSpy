#!/bin/bash
# Run a PsychoPy script with the PsychoPy environment made by install.sh
RTPSPY_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
PSY_PY="$RTPSPY_DIR/.venv-psychopy/bin/python"
if [ ! -x "$PSY_PY" ]; then
    echo "PsychoPy environment not found: run $RTPSPY_DIR/install.sh" >&2
    exit 1
fi
exec "$PSY_PY" "$@"
