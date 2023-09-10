#!/bin/bash

python3 test_titanic.py
gcc test_titanic.c random_forrest_classifier.c
./a.out

rm ./a.out
rm ./random_forrest_classifier.c
rm ./random_forrest_classifier.h
rm ./titanic_train_ans.csv