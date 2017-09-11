#!/bin/bash


for op in 'adam'; do for dn in 101; do for zd in 2 50; do for w in 2 5; do for d in 2 5; do for lr in 0.0001; do python 1d_mmd_gan.py --optimizer=$op --data_num=$dn --z_dim=$zd --width=$w --depth=$d --learning_rate=$lr; done; done; done; done; done; done;

