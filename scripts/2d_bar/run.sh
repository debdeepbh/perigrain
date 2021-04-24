path=scripts/2d_bar

l=10e-3
s=3e-3

# delete previous data
make clout 

python3 $path/setup.py $l $s
youngs=$(cat output/csv/youngs.csv)
make getfresh_py

# usual value
cp $path/base.conf config/main.conf
make ex2

make genplot

python3 gen_wall_reaction.py
#python3 $path/gen_data.py

begin=20
end=60
python3 $path/plot.py $begin $end $l $s

make showplot

