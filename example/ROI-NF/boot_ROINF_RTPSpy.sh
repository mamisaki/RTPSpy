#!/bin/bash
# Start the ROI-NF application with the RTPSpy environment made by install.sh
APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
RTPSPY_DIR="$(cd "$APP_DIR/../.." && pwd)"
exec "$RTPSPY_DIR/.venv/bin/python" "$APP_DIR/ROINF_RTPSpy.py" "$@"
