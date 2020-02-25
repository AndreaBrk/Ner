import pandas as pd
import numpy as np
import sys
import nltk
import pickle
import sklearn
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_precision_score
from sklearn_crfsuite.metrics import flat_classification_report




#Reading the csv file
df = pd.read_csv('dataset/dataset.csv', encoding = "ISO-8859-1")
df_test = pd.read_csv('dataset/test.csv', encoding = "ISO-8859-1")

#Displaying the unique Tags
labels = list(df['Tag'].unique())
print(labels)

#Displaying the unique Tags
labels_test = list(df_test['Tag'].unique())
print(labels_test)

#Checking null values, if any.
df.isnull().sum()
df_test.isnull().sum()

df = df.fillna(method = 'ffill')
df_test = df_test.fillna(method = 'ffill')


# This is a class te get sentence. The each sentence will be list of tuples with its tag.
class sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s : [(w, t) for w, t in zip(s['Word'].values.tolist(),
                                                       s['Tag'].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]
        
    def get_text(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent +=1
            return s
        except:
            return None

#Displaying one full sentence
getter = sentence(df)
sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sentences[0]

#Displaying one full sentence for test
getter_test = sentence(df_test)
sentences_test = [" ".join([s[0] for s in sent_test]) for sent_test in getter_test.sentences]


#sentence with its tag.
sent = getter.get_text()
sent_test = getter_test.get_text()
sentences = getter.sentences
sentences_test = getter_test.sentences

def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

X_train = [sent2features(s) for s in sentences]
y_train = [sent2labels(s) for s in sentences]

X_test = [sent2features(s) for s in sentences_test]
y_test = [sent2labels(s) for s in sentences_test]

crf = CRF(algorithm = 'l2sgd',
         c2 = sys.argv[1],
         max_iterations = 1000,
         all_possible_transitions = False)

# trainning                 
crf.fit(X_train, y_train)

# save model
pkl_filename =  sys.argv[2] + ".pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(crf, file)

#Predicting on the test set.
y_pred = crf.predict(X_test)

f1_score = flat_f1_score(y_test, y_pred, average = 'weighted')
print(f1_score)

report = flat_classification_report(y_test, y_pred)
print(report)
