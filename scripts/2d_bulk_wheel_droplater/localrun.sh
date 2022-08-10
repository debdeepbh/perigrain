
t_name='shapes' 

 #generate argument list of files with csv filenames
args=''
for std in "${stds[@]}"
do 
    args="$args ${std}"
done
echo "All input strings: $args"
python3 $path/plotcombinedtimestepdata.py $args $str_pref $t_name
