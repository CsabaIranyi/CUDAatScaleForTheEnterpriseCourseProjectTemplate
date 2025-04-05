#!/usr/bin/env bash
make clean build

make run ARGS="-input=data/lena.pgm -output=data/lena_gaussian_11x11.pgm -mask=11"
