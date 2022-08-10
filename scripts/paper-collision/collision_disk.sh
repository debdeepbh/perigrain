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
python3 $path/gentimestepdata.py
tag='elastic'
python3 $path/plottimestepdata.py $path $tag

#######################################################################
# damping

# copy the config file
cp $path/2d_collision_damping.conf config/main.conf

# run code
make ex2

## generate plot
#make genplot
#make genvid

## Plot motion
python3 $path/gentimestepdata.py
tag='damping'
python3 $path/plottimestepdata.py $path $tag

#######################################################################
tag='elastic'
cp $path/2d-collision-$tag-CurrPos.png ~/granular-draft/data/collision/
cp $path/2d-collision-$tag-vel.png ~/granular-draft/data/collision/

tag='damping'
cp $path/2d-collision-$tag-CurrPos.png ~/granular-draft/data/collision/
cp $path/2d-collision-$tag-vel.png ~/granular-draft/data/collision/
