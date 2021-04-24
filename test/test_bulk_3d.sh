make

# generate .msh files
gmsh meshdata/3d/closepacked_cube.geo -3
gmsh meshdata/3d/3d_sphere_small.geo -3

# setup
make setup

# get data
make getfresh_py

# run
make ex3

# plot
make genplot

