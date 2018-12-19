#!/bin/bash

filelist='../FEMH_data.lst'
basedir='/media/sdd/wzhao/data/FEMH_Data/'
resample_outdir="${basedir}processed/resample_8k/"
pitch_outdir="${basedir}processed/pitch/"

pitch_extractor="./histopitch"

[[ -d $resample_outdir ]] || mkdir -p $resample_outdir
[[ -d $pitch_outdir ]] || mkdir -p $pitch_outdir

for f in $(cat $filelist | awk '{print $1}')
do
    in_f="${basedir}${f}"
    out_f="${resample_outdir}${f%.wav}.8k.wav"
    pitch_f="${pitch_outdir}${f%.wav}.8k.pitch"

    [[ -d ${out_f%/*} ]] || mkdir -p ${out_f%/*}
    [[ -d ${pitch_f%/*} ]] || mkdir -p ${pitch_f%/*}

    sox $in_f --rate 8000 $out_f
    echo "processed ${out_f}"

    $pitch_extractor -in $in_f -out $pitch_f -srate 8000
    echo "wrote into ${pitch_f}"
done
