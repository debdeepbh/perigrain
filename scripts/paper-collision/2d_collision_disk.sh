#!/bin/bash
path="`dirname \"$0\"`"
# generate experiment setup
python3 $path/2d_collision_setup.py

# copy the data
make getfresh_py

#######################################################################
# no damping

# copy the config file
cp $path/2d_collision_elastic.conf config/main.conf

# run code
make ex2

## generate plot
#make genplot
#make genvid

## Plot motion
#python3 $path/2d_collision_elastic_plots.py
python3 $path/gentimestepdata.py
python3 $path/plottimestepdata.py $path

#######################################################################
# damping

# copy the config file
#cp $path/2d_collision_damping.conf config/main.conf

## run code
#make ex2

### generate plot
##make genplot
##make genvid

### Plot motion
#python3 $path/gentimestepdata.py
#python3 $path/plottimestepdata.py $path
