#!/bin/bash
# Copyright 2016, Yahoo
# Licensed under the terms of the Apache 2.0 license. See the LICENSE file in the project root for terms.

traindir=experiments/news
testdir=demo/test
modelfile=models/demo.clf

#train a new model and test using cross validation
python src/regression.py -d $traindir -f ngram

#train a model and save it to modelfile
#python src/regression.py -d $traindir --modelfile $modelfile --save -f light

#load a saved model and use it to make predictions
#python src/regression.py -d $testdir --modelfile $modelfile -f light --predict

#For feature/model debugging and analysis

#report performance of each feature group individually
#python src/regression.py -d $traindir -f light -x

