#!/bin/bash

echo TEST:random forrest
echo +DATA:test_titanic.csv
python3 test_random_forrest_titanic.py
gcc test_titanic.c random_forrest_classifier.c -D RANDOM_FORREST
./a.out

rm ./a.out
rm ./random_forrest_classifier.c
rm ./random_forrest_classifier.h
rm ./titanic_train_ans.csv

echo 
echo TEST:logistic regression
echo +DATA:test_titanic.csv
python3 test_logistic_regression_titanic.py
gcc test_titanic.c logistic_regression.c -D LOGISTIC_REGRESSION
./a.out

rm ./a.out
rm ./logistic_regression.c
rm ./logistic_regression.h
rm ./titanic_train_ans.csv