#!/bin/bash

gen=$1 #path to the feature's directory
sr=$2 #should be the sampling rate (normally 16000)
bits=$3 #should be the bit encoding (normally 16)
alpha=$4 #alpha for mc preamphasis (0.42 for 16k and 0.35 for 48k)
steps=$5 #winddow steps (80)
order=$6 #mc orders

files=$(ls $gen/*.pitch -1)

mkdir $gen/../synth
synth=$gen/../synth

for x in $files
do
y=${x##*/}
y=${y%.pitch}
excite -p $steps $gen/$y.pitch | mlsadf -m $order -a $alpha -p $steps $gen/$y.mcep | x2x +fs -o > $synth/$y.raw
rawtowav $sr $bits $synth/$y.raw $synth/$y.wav
done 