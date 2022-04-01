#!/bin/bash


for py in $(ls guitar*.py);
do
	log=${py%.*}.txt
	python3 -u $py >>  $log&
done 
