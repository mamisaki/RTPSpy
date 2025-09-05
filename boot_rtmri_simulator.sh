#!/bin/bash

. ${HOME}/*conda3/etc/profile.d/conda.sh
cd ${HOME}/RTPSpy
conda run --live-stream --name RTPSpy python  ./rtmri_simulator.py
