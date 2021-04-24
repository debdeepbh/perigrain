path=scripts/2d_pressure
# list of shapes, mind the surrounding space!
shapes=(
#disk
#n8
#n6
###n5
#n4
###n3
###pacman
###box
#plus0.8
plus0.2
)
timesteps=(
#18000	#disk
#22000	#n8
#26000	#n6
#14000	#n5
#60000	#n4
#16000	#n3
#44000 	#plus0.8
50000 	#plus0.2
)

str_pref="$path/2d_pressure_test/"
mkdir $str_pref

function run {
    for i in "${!shapes[@]}"
    do 
	shape=${shapes[i]}
	echo "#####################################################"
	echo "Now using shape: $shape"
	# generate experiment setup
	python3 $path/setup.py $shape
	# copy the data
	make getfresh_py
	# copy base conf
	cp $path/base$1.conf config/main.conf
	# edit timesteps
	echo "timesteps = ${timesteps[i]}" >> config/main.conf

	# run code
	make ex2
	## collect data
	#python3 $path/savedata.py $str_pref$shape$1'.csv'
	python3 gen_wall_reaction.py
	cp output/V.npy $str_pref$shape$1'.npy'

	# copy all.h5
	cp data/hdf5/all.h5 $str_pref$shape$1'.h5'

	# copy latest timestep
	dir=${str_pref}$shape$1
	mkdir -p $dir
	latest=$(ls output/hdf5/tc_*.h5 | tail -1)
	cp $latest $dir
	latest=$(ls output/hdf5/wall_*.h5 | tail -1)
	cp $latest $dir

	# plots
	make genplot
	make genvid
	mv output/img/*.png $dir

	# clean
	make clout
    done
}
########################################################################
## no fracture
run ''

#########################################################################
### with fracture
run '_frac'

#######################################################################
# generate argument list of files with csv filenames
args=''
for shape in "${shapes[@]}"
do 
    args="$args ${shape}"
    #args="$args ${shape}$1"
done
echo "All input strings: $args"
python3 $path/plot.py $args $str_pref $str_pref"force_plot.png"

sxiv $str_pref"force_plot.png" &
