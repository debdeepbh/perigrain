#!/bin/bash
path="`dirname \"$0\"`"
echo "path = $path"

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

	# create subdirectory
	dir=${str_pref}$std$1
	#mkdir -p $dir

	# generate experiment setup
	#python3 $path/setup.py $std >> $logfile
	# copy the data
	#make getfresh_py >> $logfile

	## copy all.h5
	cp $dir/h5/all.h5 data/hdf5/

	## copy the resume index
	#last=$(ls $dir/h5/tc_*.h5 | tail -1 | awk -F '_' '{ print $2 }' | awk -F '.' '{ print $1 }')

	last=$(ls $dir/h5/tc_*.h5 | tail -1) # Get the largest indices
	last=${last##*/} # strip the path
	last="${last%.*}" # strip the extension
	last=$(echo $last | awk -F '_' '{ print $2 }')

	## copy last tc_$last
	cp $dir/h5/tc_$last.h5 output/hdf5/
	cp $dir/h5/wall_$last.h5 output/hdf5/

	## copy the config file to resume
	cp $path/resume.conf config/main.conf
	## copy the config index
	echo "resume_ind = $last" >> config/main.conf
	

	# edit timesteps
	#echo "timesteps = ${timesteps[i]}" >> config/main.conf

	# run code
	echo 'running'
	make ex2 >> $logfile

	make genplot >> $logfile

	## resume
	#echo 'Resuming'

	##get y_max
	#python3 get_y_max.py "$wall_top_extra" > $dir/y_max

	## copy the config file
	#cp $path/resume.conf config/main.conf

	## copy wall top into the config
	#echo "wall_top = $(cat $dir/y_max)" 
	#echo "wall_top = $(cat $dir/y_max)" >> config/main.conf


	## copy the resume index
	#last=$(ls output/hdf5/tc_*.h5 | tail -1 | awk -F '_' '{ print $2 }' | awk -F '.' '{ print $1 }')
	#echo $last >> $dir/resume_ind
	#echo "resume_ind = $last" >> config/main.conf

	## resume code
	#make ex2 >> $logfile

	#make genplot >> $logfile

	# generate video
	#make genvid dev/null 2>&1 #> $logfile

	# generate wall reaction
	python3 gen_wall_reaction.py

	## collect data

	# copy npy files
	cp {output,$dir}/V.npy

	## copy latest timestep
	#dir=${str_pref}$std$1
	#mkdir -p $dir
	#latest=$(ls output/hdf5/tc_*.h5 | tail -1)
	#cp $latest $dir
	#latest=$(ls output/hdf5/wall_*.h5 | tail -1)
	#cp $latest $dir

	#mkdir $dir/h5
	# copy all.h5
	#cp {data/hdf5,$dir/h5}/all.h5 
	## copy h5 files for all timesteps
	cp output/hdf5/* $dir/h5/

	# copy plots
	mv output/img/*.png $dir/
	mv output/vid/*.mp4 $dir/
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
python3 $path/plot_force.py $args $str_pref $str_pref"force_plot$last.png"

sxiv $str_pref"force_plot.png" &


