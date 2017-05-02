#!/bin/bash

gen=$1 #directory containing binary feature files
order=$2 #mcep order +1
files=$(ls $gen -1)

mkdir $gen/binfeat
bin=$gen/binfeat

for x in $files
do

y=${x##*/}
ext="${x##*.}"
if [ "$ext" == "mcep" ]
then
	x2x +af $gen/$y > $bin/$y
else
	x2x +af $gen/$y > $bin/$y
fi

done 