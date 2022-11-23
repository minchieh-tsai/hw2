#!/usr/bon/env bash

hw2_1=https://www.dropbox.com/s/e3c36ne1v5yy1ms/log?dl=1
wget "${hw2_1}" -O ./code/p1/log/VGG16_0/log
python3 ./code/p1/predict.py -i $1 -o $2 

# input -i ./hw2_data/p1_data/val_50
# input -o ./code/p1/output