#!/bin/bash

gen=$1 #wav directory
sr=$2 #sampling rate (normally 16 [for 16k])
alpha=$3 #alpha for mc preamphasis (0.42 for 16k and 0.35 for 48k)
steps=$4 #winddow steps in samples (80)
order=$5 #mc orders

files=$(ls $gen/*.wav -1)

mkdir $gen/../analyz
analyzdir=$gen/../analyz

for x in $files
do
y=${x##*/}
y=${y%.wav}
x2x +sf < $gen/$y.wav | frame -l 400 -p $steps | window -l 400 -L 512 | mcep -l 512 -m $order -a $alpha > $analyzdir/$y.mcep
x2x +sf < $gen/$y.wav | pitch -a 1 -s $sr -p $steps -L 80 -H 320 > $analyzdir/$y.pitch
done 
