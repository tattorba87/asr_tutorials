#!/usr/bin/env bash

# Setting Paths
icefall_root_dir=$(realpath ~/dev/icefall)

# Add to python path so that icefall can be found by python
if [ -z "${PYTHONPATH+x}" ]; then
  PYTHONPATH=$icefall_root_dir
else
  PYTHONPATH=$icefall_root_dir:$PYTHONPATH
fi

export PYTHONPATH
echo "PYTHONPATH=$PYTHONPATH"
echo "icefall_root_dir=$icefall_root_dir"
