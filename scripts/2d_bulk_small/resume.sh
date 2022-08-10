path=scripts/2d_bulk_small
# generate mesh
#gmsh meshdata/3d/3d_sphere_small.geo -3

# generate experiment setup
#python3 scripts/2d_bulk_plus_setup.py

# copy the data
#make getfresh_py

#######################################################################
# no damping

# copy the config file
cp $path/resume.conf config/main.conf

# run code
make ex2

## generate data from the start
python3 gen_wall_reaction.py 1
python3 plot_wall_reaction.py

## generate picture with dotsize 1
make genplot

