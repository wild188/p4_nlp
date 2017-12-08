#!/bin/bash

#javac -cp jars/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/*.java

javac -cp jars/stanford-corenlp-3.8.0.jar:jars/stanford-corenlp-3.8.0-models.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/*.java

#javac -cp jars/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar:jars/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/Lesk.java
#javac -cp jars/stanford-corenlp-full-2017-06-09.zip:jars/stanford-corenlp-3.8.0-models.jar:jars/stanford-corenlp-3.8.0.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/Lesk.java
#echo jars/stanford-corenlp-full-2017-06-09/*
#javac -cp jars/stanford-corenlp-full-2017-06-09/*:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/*.java

#javac -cp jars/stanford-english-corenlp-2017-06-09-models.jar:jars/edu.mit.jwi_2.4.0.jar::bin/ -d bin/ src/Lesk.java
