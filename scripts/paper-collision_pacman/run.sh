#!/bin/bash
path="`dirname \"$0\"`"
# generate experiment setup
python3 $path/setup.py
# copy the data
make getfresh_py

#######################################################################
# no damping

# copy the config file
cp $path/2d_collision_elastic.conf config/main.conf

# run code
make ex2

## generate plot
make genplot
#python3 $path/plot.py
make showplot

## Plot motion
#python3 $path/gentimestepdata.py
#tag='elastic'
#python3 $path/plottimestepdata.py $path $tag

#######################################################################
#cp ouput/img/img_tc_00013.png ~/granular-draft/data/pacman_collision/

