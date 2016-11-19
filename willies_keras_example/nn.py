#--------------------------------------------------------------------
# Name:        keras_nn.py
#
# Purpose:     Train a Feedforward Neural network.
#
# Author:      Willie Boag
#--------------------------------------------------------------------


import os
import time
import numpy as np
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input
   

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
               continue
           #print 'tid:   [%s]' % tid
           #print 'uid:   [%s]' % uid
           #print 'label: [%s]' % label
           #print 'text:  [%s]' % text
           #print
           train_tweets.append(text)
           train_labels.append(label)

   # vocabulary of all words in training set
   vocab = list(set(' '.join(train_tweets).split()))

   # Data -> features
   train_X = extract_features(train_tweets, vocab)
   num_samples,input_dim = train_X.shape

   # e.g. 'positive' -> [1 0 0]
   num_classes = len(label_names)
   train_Y = np.array( [label2ind[label] for label in train_labels] )
   Y_onehots = to_categorical(train_Y, nb_classes=num_classes)

   # Fit model (AKA learn model parameters)
   classifier = create_model(input_dim, 300, 200, num_classes)
   classifier.fit(train_X,Y_onehots,batch_size=128,nb_epoch=10,verbose=1)

   #classifier.save_weights('tmp_keras_weights')
   #classifier.load_weights('tmp_keras_weights')


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
   test_X =  extract_features(test_tweets, vocab)
   test_Y = np.array( [label2ind[label] for label in test_labels] )
   test_Y_onehots = to_categorical(test_Y, nb_classes=num_classes)
   pred_prob = classifier.predict(test_X, batch_size=128, verbose=1)
   test_predictions = pred_prob.argmax(axis=1)


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


def extract_features(tweets, vocab):
   word2ind = { w:i for i,w in enumerate(vocab) }
   V = len(vocab)
   X = np.zeros((len(tweets),V))
   for i,tweet in enumerate(tweets):
       for word in tweet.split():
           if word not in word2ind:
               continue
           dim = word2ind[word]
           featureval  = 1        # indicate this feature is "on"
           X[i,dim] = featureval

   return X


def create_model(input_dim, embedding_dim, hidden_dim, output_dim):
   bow = Input(shape=(input_dim,))

   embeddings = Dense(output_dim=embedding_dim, activation='sigmoid')(bow)
   hidden = Dense(output_dim=hidden_dim, activation='sigmoid')(embeddings)
   prob = Dense(output_dim=output_dim, activation='softmax')(hidden)

   model = Model(input=bow, output=prob)

   print
   print 'compiling model'
   start = time.clock()
   model.compile(loss='categorical_crossentropy', optimizer='adam')
   #print '\tWARNING: skipping compilation'
   end = time.clock()
   print 'finished compiling: ', (end-start)
   print

   return model


if __name__ == '__main__':
   main()
