# Copyright 2016, Yahoo
# Licensed under the terms of the Apache 2.0 license. See the LICENSE file in the project root for terms.

"""
Wrapper classes to allow switching between different NLP toolkits for preprocessing, tagging, parsing, etc
"""

class NLTKPreprocessor(object): 
  """Use NLTK to return a bundle in the same format as the one CoreNLP returns""" 

  def __init__(self, homedir='./'):
    from nltk.stem.wordnet import WordNetLemmatizer
    self.ltzr = WordNetLemmatizer()

  def _get_wordnet_pos(self, tag):
    from nltk.corpus import wordnet 
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN #this is the default when POS is not known
 
  def parse(self, document): 
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    data = {'sentences': []}
    document = document.decode('ascii', 'ignore')
    for sent in sent_tokenize(document): 
      tokens = [w.decode('ascii', 'ignore') for w in word_tokenize(sent)]
      postags = [t for w,t in pos_tag(tokens)]
      lemmas = [self.ltzr.lemmatize(w,self._get_wordnet_pos(t)) for w,t in zip(tokens, postags)]
      data['sentences'].append({'tokens' : tokens, 'lemmas' : lemmas, 'pos': postags})
    return data

class StanfordPreprocessor(object): 
  
  def __init__(self, homedir='./'):
    from stanford_corenlp_pywrapper import CoreNLP
    self.corenlp = CoreNLP(
      configdict={'annotators':'tokenize, ssplit, pos, lemma, parse, ner'}, 
      output_types=['pos', 'lemma', 'parse', 'ner'], 
      corenlp_jars=[homedir+"lib/stanford-corenlp-full-2015-04-20/*"]
    )

  def parse(self, document): 
    return self.corenlp.parse_doc(document)
    
