# Copyright 2018, Oath Inc.
# Licensed under the terms of the Apache 2.0 license. See the LICENSE file in the project root for terms.

import re
import os
import sys
import pickle
from preprocess import NLTKPreprocessor,StanfordPreprocessor
from features import POSFeatures, PetersonFeatures, SubjectivityFeatures, LexicalFeatures, CaseFeatures, EntityFeatures, ConstituencyFeatures, DependencyFeatures, ReadabilityFeatures, W2VFeatures, NgramFeatures

class Featurizer: 
  """
  The Featurizer class is responsible for extracting features from a blob of text. 
  It is designed under the assumption that the blob of text represents a sentence, but this not necessary. 
  Note the performance might degrade if the code is running using larger (or smaller) untils of text. 
  """

  def __init__(self, parsecachepath='ref/featurizer.parsecache', use='all', homedir='./', reload_parses=True, cache_dump_freq=200000, preproc='stanford'):
    """Class constructor. 
		
    parsecachepath -- pathname for where to read/write the parse cache. If the path does not exist, a new cache will be created.
    use -- comma-separated list of features to use (see feature_names for options). If 'all', all of the available features will be used
    homedir -- home directory where the ref/ folder can be found
    reload_parses -- if True, will attempt to load parses from cache. Otherwise, will overwrite existing cache.
    cache_dump_freq -- frequency with which to write parses to cache
    """	

    self.homedir = homedir
	
    self.light_feature_names = ['ngram', 'case', 'readability', 'w2v']
    self.feature_names = ['pos', 'subjectivity', 'lexical', 'ngram', 'case', 'entity', 'constituency', 'dependency', 'readability', 'w2v']
    
    sys.stderr.write("Initializing featurizer\n")
    
    if use == 'light' : use = set(self.light_feature_names)
    elif use == 'all' : use = set(self.feature_names)
    else : use = set(use.split(','))

    #if using parse or entity features, override preproc option and force to use stanford
    if 'constituency' in use: 
      if not(preproc == 'stanford'): 
        sys.stderr.write('Warning: using constituency features requires use of StanfordPreprocessor. This might cause errors if the required dependencies are not installed\n')
        preproc = 'stanford' 
    if 'dependency' in use: 
      if not(preproc == 'stanford'): 
        sys.stderr.write('Warning: using dependency features requires use of StanfordPreprocessor. This might cause errors if the required dependencies are not installed\n')
        preproc = 'stanford' 
    if 'entity' in use: 
      if not(preproc == 'stanford'): 
        sys.stderr.write('Warning: using entity features requires use of StanfordPreprocessor. This might cause errors if the required dependencies are not installed\n')
        preproc = 'stanford' 

    self.use_features = self._get_feature_to_use(use)
    
    #Initialize preprocessor; Stanford parser required only if using dependency, constituency, or entity features
    if preproc == 'stanford' : self.preprocessor = StanfordPreprocessor()
    else : self.preprocessor = NLTKPreprocessor()
		
    #Load existing parse cache, or create a new one
    self.parsecache = {}
    self.parsecachepath = parsecachepath+'_'+preproc+'.pkl'
    if not(os.path.exists(self.parsecachepath)) : 
      self.parsecache = {}
    else : 
      sys.stderr.write('Loading parse cache...')
      if reload_parses :
        self.parsecache = pickle.load(open(self.parsecachepath))
      else : 
        self.parsecache = {}
      sys.stderr.write('done\n')

    #Sundry other initializations
    self.URL = re.compile("(www|(https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")
    self.EMAIL = re.compile("[^@]+@[^@]+\.[^@]+")
    self.new_parses = 0
    self.cache_dump_freq = cache_dump_freq
    
  def _get_feature_to_use(self, use): 
    use_features = []
    if 'case' in use: use_features.append(CaseFeatures('case', homedir=self.homedir))
    if 'constituency' in use: 
      use_features.append(ConstituencyFeatures('constituency', homedir=self.homedir))
    if 'dependency' in use: 
      use_features.append(DependencyFeatures('dependency', homedir=self.homedir))
    if 'entity' in use: 
      use_features.append(EntityFeatures('entity', homedir=self.homedir))
    if 'lexical' in use: 
      use_features.append(LexicalFeatures('lexical', homedir=self.homedir))
    if 'ngram' in use: 
      use_features.append(NgramFeatures('ngram', homedir=self.homedir))
    if 'pos' in use: 
      use_features.append(POSFeatures('pos', homedir=self.homedir))
    if 'readability' in use: 
      use_features.append(ReadabilityFeatures('readability', homedir=self.homedir))
    if 'subjectivity' in use: 
      use_features.append(SubjectivityFeatures('subjectivity', homedir=self.homedir))
    if 'w2v' in use: 
      use_features.append(W2VFeatures('w2v', homedir=self.homedir))
    return use_features

  def dump_cache(self) : 
    """Save any unsaved parses to cache"""
    if self.new_parses > 0 : 
      sys.stderr.write('Saving parses to cache...\n')
      pickle.dump(self.parsecache, open(self.parsecachepath, 'w'))
      sys.stderr.write('Save complete.\n')
      self.new_parses = 0
	
  def close(self) : 
    """Close the Featurizer and save parse cache if necessary"""
    self.dump_cache()
	
  def _replace_urls(self, s): 
    """Replace urls and emails with special token"""
    ret = '' 
    for w in s.split() : 
      if re.match(self.URL, w) : ret += '_url_ '
      elif re.match(self.EMAIL, w) : ret += '_email_ '
      else : ret += '%s '%w
    return ret.strip()

  def featurize(self, s, sid=None, use=None):
    """Extract all of the features for sentence

    s -- the sentence (or any text blob, but will work best if it is just a single sentence
    sid -- an identifier for the sentence, used to key into the cache
    """
    if use is None : use_features = self.use_features
    else : use_features = self._get_feature_to_use(use)

    if sid is None : sid = s
    s = self._replace_urls(s)
    if '%s'%sid in self.parsecache : 
      sent = self.parsecache['%s'%sid]
    else : 
      sent = self.preprocessor.parse(s)['sentences']
      self.parsecache['%s'%sid] = sent
      self.new_parses += 1
      if self.new_parses == self.cache_dump_freq: 
        self.dump_cache()
    features = {}
    for feats in use_features:
      f = feats.extract(sent)
      features.update(f)
    return features
