# generate mesh
gmsh meshdata/3d/3d_sphere_small.geo -3

path=scripts/3d_collision

# generate experiment setup
python3 $path/3d_collision_setup.py

# copy the data
make getfresh_py

#######################################################################
# no damping

# copy the config file
cp $path/3d_collision_elastic.conf config/main.conf

# run code
make ex3

## generate plot
#make genplot

## Plot motion
python3 $path/3d_collision_elastic_plots.py

#######################################################################
# damping

# copy the config file
cp $path/3d_collision_damping.conf config/main.conf

# run code
make ex3

## generate plot
#make genplot

## Plot motion
python3 $path/3d_collision_damping_plots.py
