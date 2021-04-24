path=scripts/2d_damping

python3 $path/setup.py
make getfresh_py

# usual value
cp $path/damping_base.conf config/main.conf
echo "damping_ratio = 0" >> config/main.conf
make ex2
python3 $path/gen_data.py $path/data_1.csv

# smaller
cp $path/damping_base.conf config/main.conf
echo "damping_ratio = 0.5" >> config/main.conf
make ex2
python3 $path/gen_data.py $path/data_2.csv

# bigger
cp $path/damping_base.conf config/main.conf
echo "damping_ratio = 1" >> config/main.conf
make ex2
python3 $path/gen_data.py $path/data_3.csv

python3 $path/gen_plots.py $path/data_1.csv $path/data_2.csv $path/data_3.csv $path/plot_disp.png $path/plot_vel.png $path/plot_acc.png

cp $path/plot_disp.png ~/gdrive/work/peridynamics/granular/data/pos_damp.png
cp $path/plot_vel.png ~/gdrive/work/peridynamics/granular/data/vel_damp.png
cp $path/plot_acc.png ~/gdrive/work/peridynamics/granular/data/acc_damp.png
