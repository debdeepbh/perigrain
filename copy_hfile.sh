#!/bin/bash

# copy h5 files from data directory to usual location, as if the simulation just ended
dir=0.2frac
cp scripts/2d_bulk_diffmesh_nogravity/data/$dir/h5/*.h5 output/hdf5/
cp scripts/2d_bulk_diffmesh_nogravity/data/$dir/h5/all.h5 data/hdf5/
