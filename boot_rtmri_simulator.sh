#!/bin/bash
# Start the RT-MRI simulator with the RTPSpy environment (install.sh)
cd "$(dirname "$(readlink -f "$0")")"
exec ./.venv/bin/python ./rtmri_simulator.py "$@"
