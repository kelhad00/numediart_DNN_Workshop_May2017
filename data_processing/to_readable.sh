#!/bin/bash

gen=$1 #directory containing binary feature files
order=$2 #mcep order +1
files=$(ls $gen -1)

mkdir $gen/../readable
readable=$gen/../readable

for x in $files
do

y=${x##*/}
ext="${x##*.}"
if [ "$ext" == "mcep" ]
then
	x2x +fa$order $gen/$y > $readable/$y
else
	x2x +fa $gen/$y > $readable/$y
fi

done 