#!/bin/bash

javac -cp jars/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/*.java

#javac -cp jars/stanford-corenlp-3.8.0.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/Lesk.java
