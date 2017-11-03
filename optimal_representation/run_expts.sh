#!/bin/bash

rm x_bars.txt
printf "Tried removing x_bars.txt."

CORR=0.5
NUM_DATA=1000
CHUNK_SIZE=200
PROPOSALS_PER_CHUNK=10
printf "\n\nCONFIG:\n  corr=$CORR,\n  num_data=$NUM_DATA,\n  chunk_size=$CHUNK_SIZE,\n  proposals_per_chunk=$PROPOSALS_PER_CHUNK"

for i in {1..2}; do
    python optimal_representation.py --corr=$CORR --num_data=$NUM_DATA --max_iter=1001 --save_iter=1000 --chunk_size=$CHUNK_SIZE --proposals_per_chunk=$PROPOSALS_PER_CHUNK --plot=1 &
done

wait

printf "\nRunning tests..."
python tests.py --corr=$CORR --num_data=$NUM_DATA --chunk_size=$CHUNK_SIZE --proposals_per_chunk=$PROPOSALS_PER_CHUNK
