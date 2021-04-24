path=scripts/2d_bulk
# generate mesh
#gmsh meshdata/3d/3d_sphere_small.geo -3

# generate experiment setup
python3 $path/setup.py

# copy the data
make getfresh_py

#######################################################################
# no damping

# copy the config file
cp $path/base.conf config/main.conf

# run code
make ex2

## generate plot
python3 $path/genplot_better_ini.py

## Plot motion
#python3 $path/3d_collision_elastic_plots.py


