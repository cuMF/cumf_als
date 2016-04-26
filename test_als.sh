#!/bin/bash
#$1 should be a directory
make clean build
mkdir $1
./main 10 0.058 1 > $1/als.10
./main 20 0.058 1 > $1/als.20
./main 30 0.058 1 > $1/als.30

./main 40 0.058 2 > $1/als.40
./main 50 0.058 2 > $1/als.50
./main 60 0.058 2 > $1/als.60

./main 70 0.058 3 > $1/als.70
./main 80 0.058 3 > $1/als.80
./main 90 0.058 3 > $1/als.90
./main 100 0.058 3 > $1/als.100

./main 110 0.058 4 > $1/als.110
./main 120 0.058 4 > $1/als.120
./main 130 0.058 10 > $1/als.130
./main 140 0.058 10 > $1/als.140
./main 150 0.058 10 > $1/als.150
./main 160 0.058 10 > $1/als.160
./main 170 0.058 10 > $1/als.170
./main 180 0.058 10 > $1/als.180
./main 190 0.058 10 > $1/als.190
./main 200 0.058 10 > $1/als.200
