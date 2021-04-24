path=scripts/2d_angle_of_repose

# setup
python3 $path/setup.py
make getfresh_py

# usual value
cp $path/base.conf config/main.conf
echo "friction_coefficient = 0.9" >> config/main.conf
make ex2
python3 $path/gen_data.py $path/data_1.csv

# plots
make genplot
