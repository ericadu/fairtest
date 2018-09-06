#!/bin/bash

N=$1
DIRPATH=$2
ROWTYPE=$3
P=$4

mkdir $DIRPATH/$P
COUNTER=0
while [ $COUNTER -lt $N ]; do
  python2.7 statistical_parity_generator.py -r $ROWTYPE -d $DIRPATH -p $P
  let COUNTER=COUNTER+1 
done