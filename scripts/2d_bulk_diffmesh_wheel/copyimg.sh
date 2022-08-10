
dirs=(
plusfrac
0.4frac
n4frac
)

dest="$HOME/granular-draft/data/compress_shapes/"
mkdir -p $dest

# initial setup
for i in "${!dirs[@]}"
do 
    dir=${dirs[i]}
    #echo "cp --parent $dir/img_tc_00001.png $dest"
    cp --parent $dir/img_tc_00001.png $dest
done


# particle column
cp --parent plusfrac/img_tc_00130.png $dest
cp --parent 0.4frac/img_tc_00095.png $dest
cp --parent n4frac/img_tc_00109.png $dest

# significant damage
cp --parent plusfrac/img_tc_00200.png $dest
cp --parent 0.4frac/img_tc_00180.png $dest
cp --parent n4frac/img_tc_00200.png $dest

