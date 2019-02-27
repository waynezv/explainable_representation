#!/bin/bash

target_dir="./recup_pyfiles"
save_dir="./target_pyfile"
for f in $(ls ${target_dir})
do
    filepath="$target_dir""/${f}"
    if grep "Alex" $filepath
    then
        echo $filepath
        # cp $filepath $save_dir
    fi
done
