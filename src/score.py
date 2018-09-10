# Copyright 2016, Yahoo
# Licensed under the terms of the Apache 2.0 license. See the LICENSE file in the project root for terms.

import sys
import pickle
import optparse
from featurizer import Featurizer
from preprocess import NLTKPreprocessor,StanfordPreprocessor

def feature_string(d): 
  return ' '.join(['%s:%s'%(f.replace(' ', '_'), v) for f,v in d.iteritems()])

def score(sent, ftzr, dv, clf): 
  xdict = ftzr.featurize(sent)
  x = dv.transform(xdict)
  fstr = feature_string(xdict) if opts.dump else ''
  return clf.predict(x)[0], fstr

def score_doc(sentences, ftzr, dv, clf): 
  scores = []
  fstrs = []
  for sent in sentences : 
    s, f = score(' '.join(sent['tokens']), ftzr, dv, clf)
    scores.append(s)
    fstrs.append(f)
  avg = sum(scores)/len(scores)
  return avg, scores, fstrs

def main(): 

  if opts.input == None: 
    docs_in = sys.stdin
  else: 
    docs_in = open(opts.input)

  if opts.output == None: 
    scores_out = sys.stdout
  else: 
    scores_out = open(opts.output, 'w')

  bundle = pickle.load(open(opts.model))
  clf = bundle['clf']
  dv = bundle['dv']
  ftzr = Featurizer(parsecachepath=opts.cache, use=opts.features)
  
  if opts.preproc == 'nltk':   
    preprocessor = NLTKPreprocessor()
  else: 
    preprocessor = StanfordPreprocessor()

  for doc in docs_in : 
    if doc.strip() == '' : 
      scores_out.write('\n')
    else: 
      if opts.nosplit: 
        avg, fstr = score(doc, ftzr, dv, clf)
        out = '%s'%avg
        if opts.dump : out += '\t%s'%fstr
      else:
        sentences = preprocessor.parse(doc)['sentences']
        avg, scores, fstrs = score_doc(sentences, ftzr, dv, clf)
        out = '%s'%avg
        if opts.sentscores : out += '\t%s'%(','.join(['%f'%s for s in scores]))
        if opts.dump : out += '\t%s'%','.join(fstrs)
      scores_out.write('%s\n'%out)
  scores_out.close()
  ftzr.close()

if __name__ == '__main__' : 

  optparser = optparse.OptionParser()
  optparser.add_option("-i", "--input", dest="input",  default=None, help="Input file, containing one document per line. (If None, will assume stdin.)")
  optparser.add_option("-o", "--output", dest="output",  default=None, help="Output file, will contain one score per line. (If None, will assume stdout.)")
  optparser.add_option("-m", "--model", dest="model",  default="models/answers.light.clf", help="The pre-trained model to use.")
  optparser.add_option("-f", "--features", dest="features",  default='light', 
    help="Feature set to use. Options are 'light' (default), 'all', or a comma-separated list of feature group names. See README for list of available feature groups.")
  optparser.add_option("-p", "--preproc", dest="preproc",  default='nltk', help="Preprocessor to use. Options are nltk or stanford.")
  optparser.add_option("-c", "--cache", dest="cache",  default='ref/predictions.parsecache', help="Location to store cache of parses.")
  optparser.add_option("--dump", dest="dump",  default=False, action="store_true", help="Print out values of features")
  optparser.add_option("--sentscores", dest="sentscores",  default=False, action="store_true", help="Print individual sentence scores as well as overeall document score")
  optparser.add_option("--nosplit", dest="nosplit",  default=False, action="store_true", help="Force treating each document as a whole, rather than splitting into sentences. This may give undesired behavior if using parse features.")
  
  (opts, _) = optparser.parse_args()
  main()
