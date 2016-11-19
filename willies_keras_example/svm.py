#--------------------------------------------------------------------
# Name:        svm.py
#
# Purpose:     Train an svm
#
# Author:      Willie Boag
#--------------------------------------------------------------------


import os
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC


label_names = ['positive', 'negative', 'neutral']
label2ind = { label:ind for ind,label in enumerate(label_names) }


def main():

   # Get data from notes
   train_tweets = []
   train_labels = []
   train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'twitter-train-gold-B.tsv')
   with open(train_file, 'r') as f:
       for line in f.readlines():
           tid,uid,label,text = line.strip().split('\t')
           if text == 'Not Available':
               continue                #print 'tid:   [%s]' % tid
           #print 'uid:   [%s]' % uid
           #print 'label: [%s]' % label
           #print 'text:  [%s]' % text
           #print
           train_tweets.append(text)
           train_labels.append(label)

   # Data -> features (encoded as sparse feature dictionaries)
   train_feats = extract_features(train_tweets)

   # feature dictionary -> sparse numpy vector
   vectorizer = DictVectorizer()
   train_X = vectorizer.fit_transform(train_feats)

   # e.g. 'positive' -> 0
   train_Y = np.array( [label2ind[label] for label in train_labels] )

   # Fit model (AKA learn model parameters)
   classifier = LinearSVC(C=0.1)
   classifier.fit(train_X, train_Y)


   # Predict on test data
   test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'twitter-test-gold-B.tsv')
   test_tweets = []
   test_labels = []
   with open(test_file, 'r') as f:
       for line in f.readlines():
           tid,uid,label,text = line.strip().split('\t')
           if text == 'Not Available':
               continue
           test_tweets.append(text)
           test_labels.append(label)
   test_feats =  extract_features(test_tweets)
   test_X = vectorizer.transform(test_feats)
   test_Y = np.array( [label2ind[label] for label in test_labels] )
   test_predictions = classifier.predict(test_X)


   # display a couple results
   print
   print 'references:  ', test_Y[:5]
   print 'predictions: ', test_predictions[:5]
   print

   # compute confusion matrix (rows=predictions, columns=reference)
   confusion = np.zeros((3,3))
   for pred,ref in zip(test_predictions,test_Y):
       confusion[pred][ref] += 1
   print ' '.join(label_names)
   print confusion
   print

   # compute P, R, and F1 of each class
   for label in label_names:
       ind = label2ind[label]
       
       tp         = confusion[ind,ind]
       tp_plus_fn = confusion[:,ind].sum()
       tp_plus_fp = confusion[ind,:].sum()


       precision = float(tp)/tp_plus_fp
       recall    = float(tp)/tp_plus_fn
       f1        = (2*precision*recall) / (precision+recall+1e-9)

       print label
       print '\tprecision: ', precision
       print '\trecall:    ', recall
       print '\tf1:        ', f1
       print


def extract_features(tweets):
   features_list = []
   for tweet in tweets:
       feats = {}
       for word in tweet.split():
           featurename = ('unigram',word) # tuple format not required
           featureval  = 1                # indicate this feature is "on"
           feats[featurename] = featureval
       features_list.append(feats)
   return features_list


if __name__ == '__main__':
   main()
