#!/bin/bash
path="`dirname \"$0\"`"
python3 $path/setup.py
make getfresh_py
cp $path/friction_oscillation.conf config/main.conf
make ex2

#python3 $path/friction_oscillation_plots.py

## Plot motion
python3 $path/gentimestepdata.py
tag='damping'
python3 $path/plottimestepdata.py $path $tag
