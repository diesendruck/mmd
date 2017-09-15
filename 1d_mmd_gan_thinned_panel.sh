#!/bin/bash


for op in 'rmsprop' 'adam' 'adagrad'; do for dn in 500 5000; do for zd in 1 20; do for lr in 0.001 0.00001; do python 1d_mmd_gan_thinned.py --optimizer=$op --starting_data_num=$dn --z_dim=$zd --learning_rate=$lr --width=5 --depth=10 --total_num_runs=200101 --save_iter=50000; done; done; done; done;

