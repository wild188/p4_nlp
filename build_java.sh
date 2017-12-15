#!/bin/bash

corenlp="jars/stanford-corenlp-3.8.0.jar"
models="jars/stanford-corenlp-3.8.0-models.jar"
jwi="jars/edu.mit.jwi_2.4.0.jar"

if [ ! -f $corenlp ]; then
    echo "corenlp file not found! check build script to ensure correct location."
    echo "expected corenlp file at: $corenlp"
    exit 1
fi

if [ ! -f $models ]; then
    echo "models file not found! check build script to ensure correct location."
    echo "expected models file at: $models"
    exit 1
fi

if [ ! -f $jwi ]; then
    echo "jwi file not found! check build script to ensure correct location."
    echo "expected jwi file at: $jwi"
    exit 1
fi

javac -cp $corenlp:$models:$jwi::bin/ -d bin/ src/*.java

#javac -cp jars/stanford-corenlp-3.8.0.jar:jars/stanford-corenlp-3.8.0-models.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/*.java