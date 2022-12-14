#!/bin/bash
path="`dirname \"$0\"`"
logfile=$path/output.log
# generate mesh
#gmsh meshdata/3d/3d_sphere_small.geo -3

stds=(
#0
#0.1
#0.2
#0.3
0.4
#0.6
#ring
#plus
#ring0.2
#ring0.4
)

#resume="no"
resume="yes"

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
	mkdir -p $dir


	# enable fracture of not
	if [ "$resume" = "no" ]
	then
	    # generate experiment setup
	    python3 $path/setup.py $std >> $logfile
	    # copy the data
	    make getfresh_py >> $logfile
	else
	    cp $dir/h5/all.h5 data/hdf5/
	fi

	# copy base conf
	#cp $path/base$1.conf config/main.conf
	cp $path/base.conf config/main.conf

	# read wheel rad
	echo "wheel_rad = $(cat output/wheel_rad)" >> config/main.conf

	# edit timesteps
	#echo "timesteps = ${timesteps[i]}" >> config/main.conf

	# enable fracture of not
	if [ "$1" = "frac" ]
	then
	    echo 'Enable fracture and self_contact.'

	    echo "enable_fracture = 1" >> config/main.conf
	    echo "self_contact = 1" >> config/main.conf
	else
	    echo 'Disable fracture and self_contact.'

	    echo "enable_fracture = 0" >> config/main.conf
	    echo "self_contact = 0" >> config/main.conf
	fi


	if [ "$resume" = "yes" ]
	then
	    ## get the last index
	    last=$(ls $dir/h5/tc_*.h5 | tail -1) # Get the largest indices
	    last=${last##*/} # strip the path
	    last="${last%.*}" # strip the extension
	    last=$(echo $last | awk -F '_' '{ print $2 }')

	    cp $dir/h5/tc_$last.h5 output/hdf5/
	    cp $dir/h5/wall_$last.h5 output/hdf5/
	    echo "do_resume = 1" >> config/main.conf
	    echo "resume_ind = $last" >> config/main.conf
	    echo "wall_resume = 1" >> config/main.conf

	    echo "# set a given paticle (index = 0) to movable (it was previously not movable)"
	    echo "set_movable_index = 0"
	    echo "set_movable_timestep = $last"
	    echo "set_stoppable_index = 0"
	    echo "set_stoppable_timestep = $last"
	else
	    echo "# set a given paticle (index = 0) to movable (it was previously not movable)"
	    echo "set_movable_index = 0"
	    echo "set_movable_timestep = 15000"
	    echo "set_stoppable_index = 0"
	    echo "set_stoppable_timestep = 15000"

	    echo "# reset particle zero position to bulk height at a given timestep"
	    echo "reset_partzero_y = 1"
	    echo "reset_partzero_y_timestep = 15000"
	fi
	

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
	make genvid dev/null 2>&1 #> $logfile

	# generate wall reaction
	#python3 gen_wall_reaction.py
	## copy npy files
	#cp {output,$dir}/V.npy

	#if [ "$1" = "frac" ]
	#then
	    #python3 gen_damage_data.py
	#fi 
	

	## copy latest timestep
	#dir=${str_pref}$std$1
	#mkdir -p $dir
	#latest=$(ls output/hdf5/tc_*.h5 | tail -1)
	#cp $latest $dir
	#latest=$(ls output/hdf5/wall_*.h5 | tail -1)
	#cp $latest $dir

	mkdir $dir/h5
	# copy all.h5
	cp {data/hdf5,$dir/h5}/all.h5 
	## copy h5 files for all timesteps
	cp output/hdf5/* $dir/h5/

	# copy plots
	mv output/img/*.png $dir/
	mv output/vid/*.mp4 $dir/

	sxiv $dir/*.png &
    done
}

# call function
run 'frac'
#run ''


stds=(
0.4
ring
plus
)

function velplot {
    #generate argument list of files with csv filenames
    args=''
    for std in "${stds[@]}"
    do 
	args="$args ${std}$1"
    done
    echo "All shapes to input: $args"
    python3 $path/velplot.py $args $str_pref
}

# call function
velplot 'frac'
