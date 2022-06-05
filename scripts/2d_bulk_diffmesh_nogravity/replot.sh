#!/bin/bash
path="`dirname \"$0\"`"
logfile=$path/output.log

## To produce plots with and without fracture, turn of `run` and `run frac`
stds=(
0.4frac
plusfrac
n4frac
ringfrac
)


# identifying filename prefix for h5 and png file generated
# a name for the collection  defined in stds
t_name='shapes' 
#t_name='roundness'

function wallplot {
     #generate argument list of files with csv filenames
    args=''
    for std in "${stds[@]}"
    do 
	# copy the timestep data
	cp $path/data/${std}/h5/timestep_data.h5 $path/data/timestep_${std}.h5
	args="$args ${std}"
    done

    #echo "All input strings: $args"
    #python3 $path/plotcombinedtimestepdata.py $args $str_pref $t_name
}


wallplot ''
