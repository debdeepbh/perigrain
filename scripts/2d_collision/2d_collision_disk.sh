path='scripts/2d_collision'
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
make genplot
make genvid

## Plot motion
python3 $path/2d_collision_elastic_plots.py

#######################################################################
# damping

# copy the config file
cp $path/2d_collision_damping.conf config/main.conf

# run code
make ex2

## generate plot
make genplot
make genvid

## Plot motion
python3 $path/2d_collision_damping_plots.py
