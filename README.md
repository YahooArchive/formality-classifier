# Formality Classifier

> A Python library to predict the formality level of text. 

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
  - [Predicting Formality Scores](#predicting-formality-scores)
  - [Retraining the Model](#re-training-the-model)
- [Contribute](#contribute)
- [License](#license)

## Background

Formality Classifier is a python library that uses Python NLTK (Natural Language Toolkit) and GenSim to predict the formality level of text. It also contains the source code necessary to modify and retrain the model. 

## Install

First, install NLTK:

    $ pip install nltk
    
You might also need to install the wordnet corpus for NLTK:

    $ python -m nltk.downloader wordnet
  
Then, install GenSim (python deep learning toolkit for interfacing with word2vec):

    $ pip install gensim

You will also need to download the word2vec pretrained vectors and put them in lib/. You can [get them here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing); this file is large so this could take a while.

Finally, given an input file with one sentence per line, you can run
    
    $ python src/score.py -i INPUT_FILE -o OUTPUT_FILE 

    or alternatively

    $ cat INPUT_FILE | python src/score.py > OUTPUT_FILE 

To test, run the following command and check to ensure it produces the desired output.

    $ cat demo/test/labels | python src/score.py | head
    Initializing featurizer
    Loading w2v...done
    Loading parse cache...done
    Saving parses to cache...
    -0.842802501014
    -1.61250707521
    -1.07275203467
    -0.0589734827674
    -1.59795911305
    -1.1511505327
    -1.38744717876
    -1.57112033648
    -0.497183507984
    Save complete.

**NOTES: **
- Run "python src/score.py --help" for a full list of options
- Loading w2v takes a while, but it's faster after it's initialized
- You may get the following error, but it should be safe to ignore: `Couldn't import dot_parser, loading of dot files will not be possible.`

## Usage

### Predicting Formality Scores

#### Input and output formats

The scoring script, score.py, expects input that is one document per line. By default, it will produce one score per line (i.e. per document). For best performance, a document should correspond to a single sentence, since this is how the model was trained. If documents contain multiple sentences, score.py will produce a score for each sentence individually and report the average as the document score. Optionally, you can force score.py to treat the entire document as a single unit (no sentence splitting), but if you are using more than light features (see next section), this may not give the desired behavior. 

**Options: **
  
    -i, --input FILENAME : the file to score (one document per line); if None, will assume stdin
    -o, --output FILENAME : the file to write to (one score per line); if None, will assume stdout
    --sentscores : print the list of individual sentence scores, as well as the mean (document) score. 
    --dump FNAMES : print the actual values of the features in the groups in FNAMES; if None, will print all features. FNAMES should be a comma-separated (no spaces) list of feature group names

#### Features and feature groups
  
**Options: **
  
    -f, --features FEATURES: list of features to use.  FEATURES should be 'light', 'all', or a comma-separated list of feature group names (details below).

**Feature groups**

The following feature groups are available. 

- case: capitalization and punctuation features
- constituency: productions and depth of constituency parse
- dependencies: tuples from dependency graph of sentence
- entity: numbers and types of entities in sentence
- lexical: average word length, average word log frequency, etc.
- ngrams: 1,2, and 3 grams as sparse features
- pos: frequencies of POS tags in sentence
- readability: length, complexity, and readability scores
- subjectivity: subjectivity and polarity scores, hedge words, and 1st/3rd person features
- w2v: averaged word2vec vectors for the words in the sentence

There are two sets of pre-trained models that provide combinations of the features above. The light feature set - containing case, ngrams, readability, and w2v - and the all feature set - containing all of the listed feature groups. You may use any subset or combination of the feature groups, but if you want to use a set other than 'light' or 'all', you will need to retrain the model (see below).   

**Light features**

By default, the classifier runs with the light feature set. In practice, this has performed comparably to using all features, but it requires less preprocessing and fewer dependencies. With light features we achieve the following performances on each of our 4 genres (using Spearman rho with human labels, obtained on cross validation):

-  answers:	0.72
-  emails:	0.75
-  news:	0.46
-  blogs:	0.63

Using light features requires NLTK preprocessing, and has no additional dependencies other than what is listed in the Quick Start of this README.

**All features**

Optionally, you may want to run with a feature set that uses heavier NLP features, such as parse trees and entity recognizers. In practice, we have not found the more advanced features to notably improve performance over the light feature set. With full features, we've achieved the following performance (Spearman rho) on cross validation: 

- answers:	0.70
- emails:	0.76
- news:		0.49
- blogs:	0.65

Using all features requires Stanford preprocessing, and has some additional dependencies, described below.  

#### Preprocessing Options

There are two available preprocessors: NLTKPreprocessor (default) and StanfordPreprocessor. Both perform sentence splitting, tokenization, POS tagging, and lemmatization. The StanfordPreprocessor also performs parsing and named-entity recognition. The StanfordPreprocessor is require if you are using entity, dependency, or constituency feature groups. Otherwise, NLTKPreprocessor is sufficient. However, it is possible to optionally use the StanfordPreprocessor even when it is not required.
  
**Options: **
  
    -p, --preproc nltk|stanford: preprocessor to use 
  
Note that if you try to force the use of the NLTKPreprocessor such as when Stanford preprocessing is required (e.g by using the options --features all --preproc nltk), your --preproc option will be overridden. 

Installing dependencies for StanfordPreprocessor:
    
This section is a work in progress. In short you will need to [install the stanford_corenlp_pywrapper](https://github.com/brendano/stanford_corenlp_pywrapper).

You will also need to [download the stanford corenlp jars](http://nlp.stanford.edu/software/corenlp.shtml#Download) and put them in lib/.


### Re-Training the Model

This section is a work in progress; it includes information about code structure and how to modify and retrain the model. For examples of running and training the model, see regress.sh.

#### Adding features

To add a new feature, see src/features.py. You should define a class that impliments an extract() method which takes as input a list of processed sentences (see below) and outputs a dictionary of features in the format {key: value}.

#### Adding preprocessors

To add a new preprocessor, see src/preprocess.py. The class should take a plain text string as input and return a json-style object with the following format.

      {'sentences': 
        [{'tokens': [], 'lemmas': [], 'pos': []},...]
      }

At a minimum, the preprocessor should return 'tokens', 'lemmas', and 'pos'. It can optionally contain more markup, e.g. 'parse', 'deps', 'entity', etc.

## Contribute

This project has been posted for archival purposes and is not being actively developed.

## License
This project is licensed under the terms of the [Apache 2.0](LICENSE-Apache-2.0) open source license. Please refer to [LICENSE](LICENSE) for the full terms.







