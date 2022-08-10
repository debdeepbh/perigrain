path=scripts/2d_bulk
# generate mesh
#gmsh meshdata/3d/3d_sphere_small.geo -3

# generate experiment setup
python3 $path/setup_usual_gravity.py

# copy the data
make getfresh_py


# copy the config file
cp $path/resume.conf config/main.conf

# run code
make ex2

## generate data from the start
python3 gen_wall_reaction.py 1
python3 plot_wall_reaction.py 201

## generate picture with dotsize 1
#make genplot
python3 $path/genplot_better.py
