#!/bin/bash
start=$(date +%H:%M:%S)
args=""
s=" "
 for f in `ls data/semcor_txt/*.txt`
 do
    args+=" "
    args+=$f
 done
java -cp jars/stanford-corenlp-3.8.0.jar:jars/stanford-corenlp-3.8.0-models.jar:jars/edu.mit.jwi_2.4.0.jar:../../../../data\ mining/tools/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar:bin/ Lesk $args #> results/metrics.txt
echo $start
date +%H:%M:%S