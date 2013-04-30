#!/bin/bash
for ALGO in svm knn rf lr pr
do
	python grid_search.py $1 $ALGO $2
	python kfold.py $1 $ALGO $2
done
