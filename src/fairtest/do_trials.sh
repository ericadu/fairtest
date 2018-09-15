#!/bin/bash

N=$1
DIRPATH=$2
SETTINGS=$3

while IFS='' read -r line || [[ -n "$line" ]]; do
  COUNTER=0
  while [  $COUNTER -lt $N ]; do
     python2.7 audit.py -d $DIRPATH -s $line
     let COUNTER=COUNTER+1 
  done
  python2.7 parse_results.py -d $DIRPATH -s $line
done < "$SETTINGS"