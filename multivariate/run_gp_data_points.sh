#!/bin/bash

# This script runs the MMD-based generative network for multivariate data, and
# then computes data weights based on either support points or their nearest
# neighbors in the data (thus, a coreset).
#   SNAP is {0, 1}, and defines whether to use the coreset.
#   M is the size of the support set or coreset.
#   DATA_FILE is the name of the file containing original data to be modeled.

SNAP=1
M=50
DATA_FILE='gp_data.txt'

python multivariate_mmd_gan.py --batch_size=$M; python weighting.py --snap=$SNAP;
