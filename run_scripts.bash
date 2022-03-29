#!/bin/bash


for py in $(ls guitar*.py);
do
	log=${py%.*}.txt
	python -u $py >>  $log&
done 
