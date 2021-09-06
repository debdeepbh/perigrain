#!/bin/bash

# copy h5 files from data/ directory to usual location, as if the simulation just ended

#dir=0.2frac

path="`dirname \"$0\"`"

cp $path/data/$1/h5/*.h5 output/hdf5/
cp $path/data/$1/h5/all.h5 data/hdf5/
