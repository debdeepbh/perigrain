# generate mesh
gmsh meshdata/3d/3d_sphere_small.geo -3
gmsh meshdata/3d/closepacked_cube.geo -3

# generate experiment setup
python3 scripts/3d_bulk_sphere/3d_bulk_sphere_setup.py

# copy the data
make getfresh_py

#######################################################################
# no damping

# copy the config file
cp scripts/3d_bulk_sphere/3d_bulk_sphere.conf config/main.conf

# run code
make ex3

## generate plot
make genplot

