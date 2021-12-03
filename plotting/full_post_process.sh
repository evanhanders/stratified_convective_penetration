#!/bin/bash

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
DIR=""
NCORE=""

while getopts ":d:n:h?:" opt; do
    case "$opt" in
    h|\?)
        echo "specify dir with -d and core number with -n" 
        exit 0
        ;;
    d)  DIR=$OPTARG
        ;;
    n)  NCORE=$OPTARG
        ;;
    esac
done
echo $DIR
echo $NCORE
echo "Processing $DIR on $NCORE cores"

#mpirun -n $NCORE python3 plot_rolled_structure.py $DIR
#mpirun -n $NCORE python3 plot_slices.py $DIR
mpiexec_mpt -n $NCORE python3 plot_rolled_structure.py $DIR
mpiexec_mpt -n $NCORE python3 profile_post_process.py $DIR
mpiexec_mpt -n $NCORE python3 plot_slices.py $DIR

cd $DIR
$OLDPWD/png2mp4.sh rolled_structure/ rolled_structure.mp4 30
$OLDPWD/png2mp4.sh snapshots/ snapshots.mp4 30
$OLDPWD/png2mp4.sh post_profiles/ post_profiles.mp4 30
cd $OLDPWD
