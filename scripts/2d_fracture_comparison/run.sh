path=scripts/2d_fracture_comparison

python3 $path/setup.py 1
make getfresh_py

# usual value
cp $path/base.conf config/main.conf
make ex2

make genplot
