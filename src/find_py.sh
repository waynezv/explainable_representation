#!/bin/bash

basename="./recup_dir."
destination="./recup_pyfiles"
for i in $(seq 1 921)
do
    dirname="$basename""$i"
    echo "doing ${dirname}"
    for f in $(ls $dirname)
    do
        ext="${f##*.}"
        if [ "$ext" = "py" ]
        then
            cp "$dirname""/${f}" $destination
            echo "done cp ${f}"
        fi
    done
done
