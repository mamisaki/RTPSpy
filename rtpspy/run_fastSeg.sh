#!/bin/bash

# activate anaconda environment
. $HOME/*conda3/etc/profile.d/conda.sh
conda activate fastsurfer_gpu

# Run fastSeg
./fast_seg.py $@
