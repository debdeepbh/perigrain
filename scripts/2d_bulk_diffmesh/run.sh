path=scripts/2d_bulk_diffmesh
logfile=$path/output.log
# generate mesh
#gmsh meshdata/3d/3d_sphere_small.geo -3

stds=(
0
0.2
#0.4
#0.6
)

# while resuming leave this much extra on top of the bulk
wall_top_extra='0.5e-3'

#timesteps=(
##18000	#disk
##22000	#n8
##26000	#n6
##14000	#n5
##60000	#n4
##16000	#n3
##44000 	#plus0.8
#50000 	#plus0.2
#)

str_pref="$path/data/"
mkdir $str_pref

#clear logfile
echo '' > $logfile

function run {
    for i in "${!stds[@]}"
    do 

	# clean
	make clout

	std=${stds[i]}
	echo "#####################################################"
	echo "Now using std: $std"
	# generate experiment setup
	python3 $path/setup.py $std >> $logfile
	# copy the data
	make getfresh_py >> $logfile
	# copy base conf
	#cp $path/base$1.conf config/main.conf
	cp $path/base.conf config/main.conf

	# edit timesteps
	#echo "timesteps = ${timesteps[i]}" >> config/main.conf

	# run code
	echo 'running'
	make ex2 >> $logfile

	make genplot >> $logfile


	## resume
	echo 'Resuming'

	#get y_max
	python3 get_y_max.py "$wall_top_extra" > $str_pref$std$1'_y_max'

	# copy the config file
	cp $path/resume.conf config/main.conf

	echo "wall_top = $(cat $str_pref$std$1'_y_max')" 
	#speed_wall_top = -0.5
	echo "wall_top = $(cat $str_pref$std$1'_y_max')" >> config/main.conf

	# resume code
	make ex2 >> $logfile

	make genplot >> $logfile

	## collect data
	#python3 $path/savedata.py $str_pref$shape$1'.csv'
	python3 gen_wall_reaction.py
	cp output/V.npy $str_pref$std$1'.npy'

	# copy all.h5
	echo "Copying all.h5 to: $str_pref$std$1"
	cp data/hdf5/all.h5 $str_pref$std$1'.h5'

	## copy latest timestep
	#dir=${str_pref}$std$1
	#mkdir -p $dir
	#latest=$(ls output/hdf5/tc_*.h5 | tail -1)
	#cp $latest $dir
	#latest=$(ls output/hdf5/wall_*.h5 | tail -1)
	#cp $latest $dir

	## copy all h5
	cp output/hdf5/* $str_pref$std$1

	# plots
	make genplot >> $logfile
	make genvid dev/null 2>&1 #> $logfile
	mv output/img/*.png $dir
	mv output/vid/*.mp4 $dir

    done
}

# call function
run ''

# generate experiment setup
#python3 $path/setup.py 0.2

## copy the data
#make getfresh_py

#######################################################################
 #generate argument list of files with csv filenames
args=''
for std in "${stds[@]}"
do 
    args="$args ${std}"
    #args="$args ${shape}$1"
done
echo "All input strings: $args"
python3 $path/plot_force.py $args $str_pref $str_pref"force_plot.png"

sxiv $str_pref"force_plot.png" &

