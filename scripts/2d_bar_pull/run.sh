path=scripts/2d_bar_pull

l=10e-3
s=3e-3
Force=1e4

# delete previous data
make clout 

python3 $path/setup.py $l $s $Force
make getfresh_py

# usual value
cp $path/base.conf config/main.conf
make ex2

#make genplot

python3 $path/gen_data.py $l $s $Force

