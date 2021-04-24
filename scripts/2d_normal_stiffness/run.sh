path=scripts/2d_normal_stiffness
python3 $path/setup.py
make getfresh_py

# usual value
cp $path/stiffness_base.conf config/main.conf
echo "normal_stiffness = 1.145915590261646e+22" >> config/main.conf
make ex2
python3 $path/gen_data.py $path/data_1.csv

# smaller
cp $path/stiffness_base.conf config/main.conf
echo "normal_stiffness = 1.145915590261646e+21" >> config/main.conf
make ex2
python3 $path/gen_data.py $path/data_2.csv

# bigger
cp $path/stiffness_base.conf config/main.conf
echo "normal_stiffness = 1.145915590261646e+23" >> config/main.conf
make ex2
python3 $path/gen_data.py $path/data_3.csv

python3 $path/gen_plots.py $path/data_1.csv $path/data_2.csv $path/data_3.csv $path/plot_disp.png $path/plot_vel.png $path/plot_acc.png

cp $path/plot_disp.png ~/gdrive/work/peridynamics/granular/data/pos_nodamp.png
cp $path/plot_vel.png ~/gdrive/work/peridynamics/granular/data/vel_nodamp.png
cp $path/plot_acc.png ~/gdrive/work/peridynamics/granular/data/acc_nodamp.png
