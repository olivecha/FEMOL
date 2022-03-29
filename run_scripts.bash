#!/bin/bash


for py in $(ls guitar*.py);
do
	log=${py%.*}.txt
	python $py >>  $log&
done 
