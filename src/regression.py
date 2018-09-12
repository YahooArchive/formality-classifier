# Copyright 2018, Oath Inc.
# Licensed under the terms of the Apache 2.0 license. See the LICENSE file in the project root for terms.


#experiments/by-genre/blogs/predictions/ridge.predictioni!/bin/python

import sys
import random
import pickle
import optparse
import numpy as np
from scipy.sparse import vstack
import pdb
import json
from sklearn.preprocessing import normalize,scale
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.grid_search import GridSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from featurizer import Featurizer

def batch(iterable, n = 1):
  l = len(iterable)
  for ndx in range(0, l, n):
    yield iterable[ndx:min(ndx+n, l)]

#read data and features from file
def get_data(label_file, ftzr, dv=None, encodeY=True, use_features=None): 
  sys.stderr.write("Getting data...")
  nm = []
  y = []
  X = []
  for i,labels in enumerate(open(label_file).readlines()):
    w, l = labels.strip().split('\t')
    x = ftzr.featurize(w, use=use_features)
    nm.append(w)
    if l == 'unk' : y.append(0)
    else : y.append(float(l))
    X.append(x)
  if dv is None : 
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(X)
  else : 
    X = dv.transform(X)
  sys.stderr.write(".done\n")
  return y, X, dv, nm

#train and multinomial naive bayes classifier
def train_classifier(X, Y,use_params=None):
  _clf = Ridge()
  params = {'alpha' : [0.001, 0.01, 0.1, 1,10,100,1000]}
  if use_params is not None :  
    params = {k : [use_params[k]] for k in use_params}
  clf = GridSearchCV(_clf, params, refit=True)
  clf.fit(X,Y)
  chosen_params = clf.best_params_
  if use_params is None : sys.stderr.write('Grid search selected parameters: %s\n'%str(chosen_params))
  return clf, chosen_params

#test the classifier
def test_classifier(clf, X, Y):
  return clf.score(X,Y)

def xval(X, Y, nm, dv, extra=None, printing=True, c=1.0, extra_amount=None) : 
  if extra is not None : Yp, Xp, nmp = extra
  scores = []
  seeds = [539, 990, 980, 99, 123, 675, 102, 740, 333, 694, 592, 774, 435, 930, 977, 198, 808, 188, 858, 132]#
  for num in range(10) : 
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=seeds[num])
    if extra is not None : 
      if extra_amount is not None : 
        K = int(extra_amount * X.shape[0])
      else : 
        K = int(Xp.shape[0])
      rows_idxs = random.sample(xrange(Xp.shape[0]), K)
      rows_to_add = Xp[rows_idxs, :]
      x_train = vstack((x_train, rows_to_add))
      y_train = y_train + [Yp[i] for i in rows_idxs]
    clf, params = train_classifier(x_train, y_train)

    preds = [clf.predict(x)[0] for x in x_test]
    s, p = spearmanr(preds, y_test)
		
    if printing: print 'Fold %d:\t%s\t%s'%(num, len(y_test), s)
    scores.append(s)
  print 'Average:\t', (sum(scores) / len(scores))

def predict_labels(clf, X, nm) : 
  for x,n in zip(X,nm) : 
    pred = clf.best_estimator_.predict(x)[0]
    print '%s\t%s'%(n, pred)
			

if __name__ == '__main__' : 

  optparser = optparse.OptionParser()
  optparser.add_option("-d", "--dir", dest="dir",  default="data/", help="Root data directory")
  optparser.add_option("-f", "--features", dest="features",  default='light', help="Comma separated list of feature groups to use")
  optparser.add_option("-v", "--save", dest="save",  default=False, action="store_true", help="Train a model and save it to the specified file.")
  optparser.add_option("-m", "--modelfile", dest="modelfile", help="File to read model from/write model to.")
  optparser.add_option("-p", "--predict", dest="predict",  default=False, action="store_true", help="Load a saved mode and use it to on unseen data.")
  optparser.add_option("-x", "--feature_selection", dest="feature_selection",  default=False, action="store_true", help="Print performance of feature groups one at a time.")
  optparser.add_option("-e", "--extra_train", dest="extra_train",  default=None, type="string", help="Add extra (possibly out of domain) data to training")
  optparser.add_option("-a", "--ablation", dest="ablation",  default=False, action="store_true", help="Run ablation analysis by feature group.")
  optparser.add_option("-r", "--print_best_features", dest="print_best_features",  default=False, action="store_true", help="Print features with highest weights.")
	
  (opts, _) = optparser.parse_args()

  label_file = "%s/labels"%opts.dir
  ftzr = Featurizer(use=opts.features)

  if opts.predict : 
    bundle = pickle.load(open(opts.modelfile))
    clf = bundle['clf']
    dv = bundle['dv']
    _, X, _, nm = get_data(label_file, ftzr, dv, encodeY=False)

  else : 
    y, X, dv, nm = get_data(label_file, ftzr)
    yplus = None
    Xplus = None
    nmplus = None
    if opts.extra_train is not None : 
      for dr in opts.extra_train.split(',') : 
        label_file = "%s/labels"%dr
        _y, _X, _, _nm = get_data(label_file, ftzr, dv, use_features=opts.features) #use the same dv; not perfect but good enough
        if yplus is None : 
          yplus = _y
          Xplus = _X
          nmplus = _nm
        else : 
          yplus = yplus + _y
          Xplus = vstack((Xplus, _X))
          nmplus = nmplus + _nm

  if opts.save:
    clf, params = train_classifier(X, y)
    print "Training accuracy: ", test_classifier(clf, X, y)
    bundle = {'clf' : clf, 'dv' : dv, 'params' : params}
    pickle.dump(bundle, open(opts.modelfile, 'w'))

  elif opts.predict:
    predict_labels(clf, X, nm)
 
  elif opts.print_best_features: 
    nrows, ncols = X.shape
    clf, params = train_classifier(X, y)
    for i,c in sorted(enumerate(clf.best_estimator_.coef_), key=lambda e: e[1], reverse=True )[:50]: 
      print '%s\t%f'%(dv.feature_names_[i], c)

  elif opts.ablation : 
      if opts.features == 'all' : feat_groups = ftzr.feature_names
      elif opts.features == 'light' : feat_groups = ftzr.light_feature_names
      else: feat_groups = opts.features.split(',')
      for i in range(len(feat_groups)) : 
        fg = feat_groups[:i] + feat_groups[i+1:]
        print "Omitting", feat_groups[i], '----------------'
        y, X, dv, nm = get_data(label_file, ftzr, use_features=','.join(fg))
        xval(X, y, nm, dv)
        print '-----------------------------\n'
   
  elif opts.feature_selection: 
      if opts.features == 'all' : feat_groups = ftzr.feature_names
      elif opts.features == 'light' : feat_groups = ftzr.light_feature_names
      else: feat_groups = opts.features.split(',')
      for i in range(len(feat_groups)) : 
        fg = feat_groups[i]
        print "Using only", feat_groups[i], '----------------'
        y, X, dv, nm = get_data(label_file, ftzr, use_features=fg)
        xval(X, y, nm, dv)
        print '-----------------------------\n'

  else : #just run xval by default
    if opts.extra_train : 
#     for a in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] : #tune amount of additional training data used
      for a in [1.0] : 
        xval(X, y, nm, dv, (yplus, Xplus, nmplus), extra_amount=a)
    else : xval(X, y, nm, dv)
  ftzr.close()
