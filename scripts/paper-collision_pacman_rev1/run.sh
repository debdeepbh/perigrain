#!/bin/bash
path="`dirname \"$0\"`"

# generate experiment setup
#python3 $path/setup.py
python3 $path/setup.py $1 $2 $3

savedir="$HOME/img_pacman_comp/$1_$2_$3"

#######################################################################
# copy the data
make getfresh_py

# copy the config file
cp $path/2d_collision_elastic.conf config/main.conf

# save files and config
cp $path/2d_collision_elastic.conf output/img/
cp $path/setup.py output/img/

# run code
make ex2

## generate plot
#make genplot
python3 $path/plot_current.py

## copy
mkdir -p $savedir
cp output/img/* $savedir/

# copy h5 files
cp output/hdf5/* $savedir/
cp data/hdf5/all.h5 $savedir/setup.h5

#python3 $path/plot.py
#make showplot

## Plot motion
#python3 $path/gentimestepdata.py
#tag='elastic'
#python3 $path/plottimestepdata.py $path $tag

#######################################################################
#cp ouput/img/img_tc_00013.png ~/granular-draft/data/pacman_collision/

