#!/bin/bash

SNAP=1
M=200
DATA_FILE='gp_data.txt'

python multivariate_mmd_gan.py --data_num=$M; python weighting.py --snap=$SNAP;
