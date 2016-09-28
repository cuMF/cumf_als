#!/bin/bash
echo -e "F\tRMSE\t\tALStime\truntime"
FILES=$1/*
for f in $FILES
do
  #echo "Processing $f ..."
  # take action on each file. $f store current file name
  ALStime=$(cat $f |grep "update"|grep "run"|grep "gridSize"|grep -v kernel|awk '{sum += $4} END {print sum}')
  runtime=$(cat $f |grep 'doALS takes'|awk '{print $4}')
  RMSE=$(cat $f |grep 'Test RMSE in iter 9'|awk '{print $7}')
  F=$(cat $f |grep 'F = '|grep 'lambda = ' |awk '{print $9}'|awk -F',' '{print $1}')
  echo -e "$F\t$RMSE\t$ALStime\t$runtime"

done
