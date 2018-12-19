#!/bin/bash

basename="Training_Dataset"
dirname0="Normal"
dirname1="Pathological/Neoplasm"
dirname2="Pathological/Phonotrauma"
dirname3="Pathological/Vocal_palsy"

normal_folder="$basename""/""$dirname0"
pathol_folder1="$basename""/""$dirname1"
pathol_folder2="$basename""/""$dirname2"
pathol_folder3="$basename""/""$dirname3"

list_file="FEMH_data.lst"

[[ ! -e $list_file ]] || rm $list_file
touch $list_file

for f in $(ls $normal_folder | sort -V)
do
    echo "$normal_folder""/""$f" "0" >> $list_file
done

for f in $(ls $pathol_folder1 | sort -V)
do
    echo "$pathol_folder1""/""$f" "1" >> $list_file
done

for f in $(ls $pathol_folder2 | sort -V)
do
    echo "$pathol_folder2""/""$f" "2" >> $list_file
done

for f in $(ls $pathol_folder3 | sort -V)
do
    echo "$pathol_folder3""/""$f" "3" >> $list_file
done

