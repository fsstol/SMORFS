#!/bin/bash
FLAGS="-std=c++20 -shared -fPIC -O3 -march=native"
# the Fourier series' matrix product through a BLAS: Accelerate on macOS (elsewhere Eigen's own product)
if [ "$(uname)" = "Darwin" ]; then FLAGS="$FLAGS -DSMORFS_USE_BLAS -framework Accelerate"; fi
g++ -o semi_markov.so src/src_lib/library.cpp $FLAGS
