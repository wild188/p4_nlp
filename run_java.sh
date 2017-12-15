#!/bin/bash

date +%H:%M:%S

corenlp="jars/stanford-corenlp-3.8.0.jar"
models="jars/stanford-corenlp-3.8.0-models.jar"
jwi="jars/edu.mit.jwi_2.4.0.jar"

if [ ! -f $corenlp ]; then
    echo "corenlp file not found! check run script to ensure correct location."
    echo "expected corenlp file at: $corenlp"
    exit 1
fi

if [ ! -f $models ]; then
    echo "models file not found! check run script to ensure correct location."
    echo "expected models file at: $models"
    exit 1
fi

if [ ! -f $jwi ]; then
    echo "jwi file not found! check run script to ensure correct location."
    echo "expected jwi file at: $jwi"
    exit 1
fi

args=`ls data/semcor_txt/*.txt`

java -cp $corenlp:$models:$jwi::bin/ Lesk $args > results/metrics.txt

date +%H:%M:%S