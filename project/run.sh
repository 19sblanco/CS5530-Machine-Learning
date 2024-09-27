#!/bin/bash

echo "running all the code by default\n"
echo "to run a certain file, go into run.sh and use comments to pick which file to run"

echo "running decision_tree0.py\n"
python3 turn_in/decision_tree0.py

echo "running ensemble_perceptron.py\n"
python3 turn_in/ensemble_perceptron.py

echo "running logistic_regression.py\n"
python3 turn_in/logistic_regression.py

echo "running perceptron.py\n"
python3 turn_in/perceptron.py

echo "running svm.py\n"
python3 turn_in/svm.py
