Billy DeLucia
wld217@lehigh.edu
CSE498 Natural Language Processing Project 4

I have implemented ALL_WORDS, ALL_WORDS_R, and WINDOW. As well as JACCARD and COSINE similarity metrics. I did not implement the POS method as this necessitated some larger changes that I did not have time for this finals period. I left an Overall.txt file in the results section that is the average precision, recall, and f1 scores from some of my test runs with different parameters. Maybe this will save you some runtime.

I was able to modify the runscript and data structures to cut the runtime down to 6 minutes. I also reformatted the data structures that were produced by the predict function to allow the evaluate function to work more efficiently. I tried to implement this in a way the would lend itself to later concurrency but ran out of time. 
