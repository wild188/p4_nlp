#!/bin/bash

for f in `ls data/semcor_txt/*.txt`
do
	# echo $f
	java -cp jars/stanford-corenlp-3.8.0.jar:jars/edu.mit.jwi_2.4.0.jar:../../../../data\ mining/tools/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar:bin/ Lesk $f > results/metrics.txt
	# break
done
