#!/bin/bash

. ${HOME}/*conda3/etc/profile.d/conda.sh
conda activate psychopy

python $@
