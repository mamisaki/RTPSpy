#!/bin/bash

. ${HOME}/*conda3/etc/profile.d/conda.sh
conda activate psychopy

echo $CONDA_DEFAULT_ENV

python $@
