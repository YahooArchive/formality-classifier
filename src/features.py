# Copyright 2016, Yahoo
# Licensed under the terms of the Apache 2.0 license. See the LICENSE file in the project root for terms.


import re
import os
import csv
import sys
import gzip
import pickle
from math import log,sqrt

class FeatureGroup(object): 
  """
  A FeatureGroup is a glorified dictionary representing a set of similar features
  """
  def __init__(self, name, homedir='./'):
    self.name = name
    self.homedir = homedir

  def fname(self, f) : return '%s_%s'%(self.name, f)

  def add(self, features, feature, value): 
    features[self.fname(feature)] = float(value)
    return features

  def inc(self, features, feature, value): 
    fnm = self.fname(feature)
    value = float(value)
    if fnm in features: 
      features[fnm] += value
    else: 
      self.add(features, feature, value) 
    return features

  def norm_by(self, features, d, exclude=[]): 
    exclude = [self.fname(f) for f in exclude]
    d = float(d)
    return {f : v/d for f,v in features.iteritems() if f not in exclude}
  
  def feature_string(self, features): 
    return ' '.join(['%s:%s'%(f,v) for f,v in features.iteritems()])

class POSFeatures(FeatureGroup) : 
  """
  FeatureGroup of POS tags and the frequencies
  """

  def __init__(self, name, homedir='./'):
    super(POSFeatures, self).__init__(name, homedir)
    #Load list of POS tags and descriptions, useful for debugging
    self.tags = {row['Tag'] : row['Description'] for row in csv.DictReader(open(self.homedir+'ref/treebank-tags.txt'),delimiter='\t')}

  def extract(self, doc) : 		
    features = {}
    for t in self.tags : features = self.add(features, t, 0.)
    L = 0.
    for sent in doc: 
      for t in sent['pos'] :
        if t in self.tags : 
          features = self.inc(features, self.tags[t], 1.)

    if L > 0 : 
      features = self.norm_by(features, L)
    return features

class PetersonFeatures(FeatureGroup):
  """
  Features reimplimented from https://github.com/burgersmoke/enron-formality/blob/master/furcoat/furcoat.py	
  """

  def __init__(self, name, homedir='./'):
    super(PetersonFeatures, self).__init__(name, homedir)
    #Load wordlist used in Peterson et al. (2011) feature set
    self.informal_words = {}
    for line in open(self.homedir+'ref/enron_wordnik.txt').readlines():
      word, wordDict = line.split('=',1)
      self.informal_words[word] = eval(wordDict) 

  def _peterson_punctuation_features(self, email) :
    self.add('Number of ...', 0.)
    self.add('Number of !', 0.)
    self.add('Number of ??', 0.)
    self.add('Missing sentence final', 0.)
    for sent in email : 
      sentstr = ' '.join(sent['tokens'])
      self.inc('Number of ...', sentstr.count('...'))
      self.inc('Number of ...', sentstr.count('. . .'))
      self.inc('Number of !', sentstr.count('!'))
      self.inc('Number of ??', sentstr.count('??'))
      if not(sentstr[-1]) in ['.', '?', '!'] : self.inc('Missing sentence final', 1)

  def _peterson_sentence_features(self, email) :
    self.add('lowercaseStart', 0.)
    self.add('lowercaseSent', 0.)
    for sent in email : 
      sentstr = ' '.join(sent['tokens'])
      words = sent['tokens']
      if len(words) > 2:
        if sentstr[0].islower() : self.inc('lowercaseStart', 1)
        if sentstr.islower() : self.inc('lowercaseSent', 1)
	
  def _peterson_word_features(self, email):
    self.add('InformalWord', 0.)
    for sent in email:		
      for word in sent['tokens'] : 
        if not word.isupper():
          word = word.lower()
          if word in self.informal_words:
            for label in self.informal_words[word]:
              # NOTE : Changing this to be 'InformalWord' with a count of 1 jumped the MaxEnt accuracy from 74% to 85% (from furcoat.py)
	      feats['InformalWord'] += 1

  def extract(self, doc): 
    features = {}
    self._peterson_punctuation_features()
    self._peterson_sentence_features()
    self._peterson_word_features()
    return features

class SubjectivityFeatures(FeatureGroup) :
  """
  Feature group capturing subjectivity, hedging, first person vs. third person, etc.
  """

  def __init__(self, name, homedir='./'):
    super(SubjectivityFeatures, self).__init__(name, homedir)
    #Load word lists for hedges and pronouns, used in subjectivity features
    self.hedges = set([w.strip() for w in open(self.homedir+'ref/hedge-words.txt').readlines()])
    self.firstperson = set([w.strip() for w in open(self.homedir+'ref/1st-person-pronouns.txt').readlines()])
    self.thirdperson = set([w.strip() for w in open(self.homedir+'ref/3rd-person-pronouns.txt').readlines()])

  def _add_dict_features(self, features, sent, name, lookup):
    for w in sent['tokens']: 
      if w in lookup: 
        features = self.inc(features, name, 1)
    return features

  def extract(self, doc): 
    from textblob import TextBlob
    features = {}
    for sent in doc: 
      L = float(len(sent['tokens']))

      #Passive voice
      postags = zip(sent['lemmas'], sent['pos'], sent['tokens'])
      for i in range(1,len(postags)) : 
        if (postags[i][1] == 'VBN') and (postags[i-1][0] == 'be') : 
          features = self.inc(features, 'Number of passive constructions', 1)

      #Hedge words and pronouns
      features = self._add_dict_features(features, sent, 'Number of hedge words per sentence', self.hedges)
      features = self._add_dict_features(features, sent, 'Number of 1st person pronouns per sentence', self.firstperson)
      features = self._add_dict_features(features, sent, 'Number of 3rd person pronouns per sentence', self.thirdperson)

      #Sentiment
      rawsent = ' '.join(sent['tokens'])
      blob = TextBlob(rawsent)
      features = self.add(features, 'Subjectivity', blob.sentiment.subjectivity)
      polarity = blob.sentiment.polarity 
      features = self.add(features, 'Positive polarity', polarity if polarity > 0 else 0)
      features = self.add(features, 'Negative polarity', polarity if polarity < 0 else 0)

    self.norm_by(features,L,exclude=['Subjectivity','Positive polarity','Negative polarity'])
    return features

class LexicalFeatures(FeatureGroup) :
  """
  Feature group capturing features of individual words, e.g. length, log frequency, and log ratio score from NAACL 2015 paper
  """

  def __init__(self, name, homedir='./'):
    from nltk.corpus import stopwords
    super(LexicalFeatures, self).__init__(name, homedir)
    #Regex for matching contrations
    self.contraction = re.compile("(.*)'(.+)")
    #List of stop words
    self.stops = set(stopwords.words('english'))
    #Load unigram counts 
    self.ngram_freq = {}
    sys.stderr.write("Loading WebIT unigram counts...")
    with gzip.open(self.homedir+'ref/google-1gms.gz') as f :
      for line in f :
        word, count = line.strip().split('\t')
        self.ngram_freq[word] = log(float(count))
    sys.stderr.write("done\n")
    #Load style scores
    sys.stderr.write("Loading LogRatio style scores...")
    self.scores = {}
    with open(self.homedir+'ref/naacl-formality-scores.txt') as f :
      for i,line in enumerate(f) :
        score, p, n, nc = line.strip().split('\t')
        if float(n) > 20 : self.scores[p] = float(score)
    sys.stderr.write("done\n") 

  def extract(self, doc):
    L = 0.
    features = {}
    for sent in doc: 
      L += float(len(sent['tokens']))
      features = self.inc(features, 'Avg. word length (in characters)', sum([len(w) for w in sent['tokens']]))
      nonstops = [w.lower() for w in sent['tokens'] if not(w.lower() in self.stops)]
      features = self.inc(features, 'Avg. word log frequency', sum([self.ngram_freq.get(w.lower(),0) for w in nonstops]))
      features = self.inc(features, 'Avg. word formality score', sum([self.scores.get(w.lower(),0) for w in sent['tokens']]))
      num_contractions = sum([1 if (re.match(self.contraction,w) and not(tag == 'POS')) else 0 for (w,tag) in zip(sent['tokens'], sent['pos'])])
      features = self.inc(features, 'Number of contractions', num_contractions)

    features = self.add(features, 'Number of words per sentence', L)
    features = self.norm_by(features, L, exclude=["Number of words per sentence"])
    return features

class NgramFeatures(FeatureGroup): 
  """
  Feature group for ngram features
  """

  def __init__(self, name, homedir='./', N=3):
    super(NgramFeatures, self).__init__(name, homedir)
    self.N = N

  def extract(self, doc):
    from nltk.util import ngrams
    features = {}
    for sent in doc:
      words = sent['tokens']
      for n in range(1,self.N):
        for ngm in ngrams(words,n): 
          features = self.add(features, ' '.join([w.lower() for w in ngm]), 1)
    return features
	
class CaseFeatures(FeatureGroup): 
  """
  Feature group for casing and punctuation features
  """
  def extract(self, doc): 
    features = {}
    for sent in doc: 
      #Casing
      caps = sum([1 if (w.isupper() and not ((w == 'I'))) else 0 for w in sent['tokens']])
      features = self.inc(features, 'Number of capitalized words', caps)
      all_lower = sum([0 if w.islower() else 1 for w in sent['tokens'] if w.isalpha()]) > 0 
      features = self.inc(features, 'All lowercase sentence', all_lower)
      features = self.inc(features, 'Lowercase initial sentence', 1 if sent['tokens'][0].islower() else 0)
     
      #Punctuation
      for punct in ['!', '?', '...'] : 
        for w in sent['tokens']: 
          if punct in w: 
            features = self.inc(features, 'Number of %s per sentence'%punct, 1)

    return features

class EntityFeatures(FeatureGroup):
  """
  Feature group for casing and punctuation features
  """
  def extract(self, doc): 
    L = 0
    features = {}
    for sent in doc: 
      name_lengths = 0.
      total_names = 0.
      this_guy = ''
      last = '0'
      for i,e in enumerate(sent['ner']): 
        #indicators for the types of entities that appear
        features = self.add(features, e,1)
        #features about how names of people are used
        if e == 'PERSON' : 
          this_guy += sent['tokens'][i] + ' '
        else : 
          if last == 'PERSON' : 
            total_names += 1
            name_lengths += len(this_guy)
            this_guy = ''
            last = e
      if total_names > 0 : 
        features = self.add(features, 'Avg. name length', name_lengths/total_names)
    return features

class ConstituencyFeatures(FeatureGroup):
  """
  Feature group for productions/depth of constituency parse tree
  """
  
  def __init__(self, name, homedir='./', N=3):
    super(ConstituencyFeatures, self).__init__(name, homedir)

  def extract(self, doc):
    from nltk.tree import Tree
    features = {}
    L = 0
    depth = 0
    for sent in doc: 
      toks = sent['tokens']
      L += float(len(toks))
      parse = Tree.fromstring(sent['parse'])
      for prod in parse.productions() : 
        pstr = str(prod)
        if not(pstr.isupper()) : continue #hack to skip over lexicalized productions
        features = self.inc(features, pstr, 1)
        depth = max(depth, parse.height())
    features = self.add(features, 'Depth of constituency parse', depth)
    features = self.norm_by(features, L)
    return features

class DependencyFeatures(FeatureGroup):
  """
  Feature group for tuples of dependecies
  """
  
  def __init__(self, name, homedir='./',lexicalized=False, backed_off=True):
    super(DependencyFeatures, self).__init__(name, homedir)
    self.lexicalized = lexicalized
    self.backed_off = backed_off

  def extract(self, doc):
    L = 0
    features = {}
    for sent in doc: 
      toks = sent['tokens']
      L += float(len(toks))
      lookup = {-1: ('ROOT', 'ROOT')}
      for i,(tok,tag) in enumerate(zip(sent['tokens'], sent['pos'])) :
        lookup[i] = (tok, tag)
      for dep in sent['deps_basic'] : 
        t, gidx, didx = dep
        g, gt = lookup[gidx]
        d, dt = lookup[didx]

        #indicators for dependency types
        features = self.add(features,t,1)

        #lexicalized dependencies
        if self.lexicalized : 
          features = self.add(features,'%s-%s-%s'%(g,t,d), 1)
          features = self.add(features,'%s-%s'%(g,t), 1)				
          features = self.add(features,'%s-%s'%(t,d), 1)
          features = self.add(features,'%s-%s'%(g,d), 1)

        #backed-off dependencies
        if self.backed_off: 
          features = self.add(features,'%s-%s-%s'%(gt,t,dt), 1)
          features = self.add(features,'%s-%s'%(gt,t), 1)				
          features = self.add(features,'%s-%s'%(t,dt), 1)
          features = self.add(features,'%s-%s'%(gt,dt), 1)
    return features

class ReadabilityFeatures(FeatureGroup) : 
  """
  Feature group for length and readability features
  """
  
  def __init__(self, name, homedir='./'):
    from nltk.corpus import cmudict
    super(ReadabilityFeatures, self).__init__(name, homedir)
    self.d = cmudict.dict()

  #from https://groups.google.com/forum/#!topic/nltk-users/mCOh_u7V8_I
  def _nsyl(self, word): 
    import curses
    from curses.ascii import isdigit
    word = word.lower()
    if word in self.d : 
      return min([len(list(y for y in x if y[-1].isdigit())) for x in self.d[word.lower()]])
    else : return 0

  def _FK(self, toks): 
    words =0.
    sents = 1.
    syllables = 0.
    for w in toks:
      words += 1
      syllables += self._nsyl(w)
    if words > 0 and sents > 0 : 
      return (0.39 * (words/sents)) + (11.8 * (syllables/words)) - 15.59
    return 0

  def extract(self, doc):
    features = {}
    alltoks = []
    for sent in doc: 
      alltoks += sent['tokens']
      features = self.inc(features, 'length in words', len(sent['tokens']))
      features = self.inc(features, 'length in characters', len(' '.join(sent['tokens'])))
    features = self.add(features, 'FK score', self._FK(alltoks))
    return features

class W2VFeatures(FeatureGroup):
  """
  Feature group for w2v features
  """
  
  def __init__(self, name, homedir='./'):
    from gensim.models import word2vec, doc2vec
    super(W2VFeatures, self).__init__(name, homedir)
    #Load word2vec pretrained vectors
    sys.stderr.write("Loading w2v...")
    self.w2v = word2vec.Word2Vec.load_word2vec_format(self.homedir+'lib/GoogleNews-vectors-negative300.bin', binary=True)
    sys.stderr.write("done\n")

  def extract(self, doc):	
    import numpy as np
    features = {}
    v = None
    d1 = None
    total = 0.
    for sent in doc: 
      d1 = doc
      for w in sent['tokens'] : 
        try : 
          wv = np.array(self.w2v[w.lower()])
          if (max(wv) < float('inf')) and (min(wv) > -float('inf')) : 
            if v is None : v = wv
            else : v += wv
            total += 1
        except KeyError : 
          continue
      if v is not None : 
        v = v / total
        for i,n in enumerate(v):
          if (n == float('inf')) : n = sys.float_info.max
          if (n == -float('inf')) : n = -sys.float_info.max
          features = self.add(features, 'w2v-%d'%i, n)
      else : 
        features = self.add(features, 'w2v-NA', 1)
    return features
