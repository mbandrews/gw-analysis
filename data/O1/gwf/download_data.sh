#!/bin/bash

segdir=segments
INPUT="segments/O1_H1_gwosc_idx002_list.txt"
#INPUT="segments/O1_L1_gwosc_idx002_list.txt"

echo "Downloading data..." > $INPUT.log
n=1
while read line; do
# reading each line
    echo "$n $line" >> $INPUT.log
    wget -q $line
    # get filename
    fname=$(basename $line)
    # mv to segments folder
    echo "moving $fname to $datadir/"
    mv $fname $datadir/
    n=$((n+1))
done < $INPUT
echo "...done." >> $INPUT.log
