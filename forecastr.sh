#!/bin/bash


# rm results.txt > /dev/null 2>&1
# rm checkpoints/* > /dev/null 2>&1
# rm logs/*/* > /dev/null 2>&1

# mkdir checkpoints > /dev/null 2>&1
# mkdir logs > /dev/null 2>&1

while getopts "t:m:" option; do
    case ${option} in
    t) methodType=${OPTARG};;
    m) runMode=${OPTARG};;
    *)
        echo $"Usage: $0 -t {pg, a3c} -m {train, forecast}"
        exit 1
    esac
done

case $methodType in
    pg) methodName="pg";;
    a3c) methodName="a3c";;
    *)
        echo $"Usage: $0 -t {pg, a3c}"
        exit 1
esac

case $runMode in
    train) script="train.py";;
    forecast) script="forecast.py";;
    *)
        echo $"Usage: $0 -m {train, forecast}"
        exit 1
esac


 python3 $script $methodName
