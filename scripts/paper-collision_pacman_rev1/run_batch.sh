#!/bin/bash
path="`dirname \"$0\"`"

mlist=(4 5 6 7)
dellist=(1 2 3 4)

#mlist=(5 10 15 20)
#dellist=(1)

rc=8
#rc=5
#rc=3

rclist=(3 5 8)

data_loc="$HOME/img_pacman_comp"


function run() {
    for m in ${mlist[@]}
    do
	for del in ${dellist[@]}
	do
	    echo "Now running: $del $m $rc"
	    $path/run.sh $del $m $rc
	done
    done
}

function show() {
    # show img
    for m in ${mlist[@]}
    do
	for del in ${dellist[@]}
	do
	    sxiv ${data_loc}/${del}_${m}_${rc}/img_tc_00025.png &
	done
    done
}

function genplot() {
    # show img
    for m in ${mlist[@]}
    do
	for del in ${dellist[@]}
	do
	    #for rc in ${rclist[@]}
	    #do
		python3 $path/plot_current.py --all_dir ${data_loc}/${del}_${m}_${rc}
	    #done
	done
    done
}

function markdown() {


    arr1=("${mlist[@]}") 
    arr2=("${dellist[@]}") 
    name1=m
    name2=del

	
    ## Rc vs delta
    #rclist=(3 5 8)
    #arr1=("${rclist[@]}") 
    #arr2=("${mlist[@]}") 
    #name1=Rc
    #name2=m


    ## markdown table
    file=$HOME/img_compare.md
    rm $file

    #echo "# comparison" >> $file
    #echo "" >> $file

    # headers
    #echo -n "| " >> $file
    echo -n "|    |" >> $file
    for val2 in ${arr2[@]}
    do
	echo -n " $name2=$val2 | "  >> $file
    done
    echo "" >> $file

    # alignment
    #echo -n "| " >> $file
    echo -n "| :---: | " >> $file
    for val2 in ${arr2[@]}
    do
	echo -n " :---: |"  >> $file
    done
    echo "" >> $file

    #for m in ${mlist[@]}
    for val1 in ${arr1[@]}
    do
	#echo -n "| " >> $file
	echo -n "| $name1=$val1 | " >> $file
	#for del in ${dellist[@]}
	for val2 in ${arr2[@]}
	do
	    #echo -n " ($val2,$val1) |" >> $file
	    echo -n " ![img](${data_loc}/${val2}_${val1}_${rc}/img_tc_00025.png) |" >> $file
	    ## Rc vs delta
	    #echo -n " ![img](${data_loc}/1_${val2}_${val1}/img_tc_00025.png) |" >> $file
	done
	echo "" >> $file
    done

    ## convert markdown to html
    html=$HOME/img_compare.html
    pandoc $file -s -o $html --css $HOME/pandoc.css
    firefox $html
}

#######################################################################

#run
#show
genplot
markdown
