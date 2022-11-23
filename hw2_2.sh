hw2_2=https://www.dropbox.com/s/fruavy3zmft8hpg/epoch_15?dl=1
wget "${hw2_2}" -O ./code/p2/log/FCN32/fcn32
python3 ./code/p2/predict32.py -i $1 -o $2