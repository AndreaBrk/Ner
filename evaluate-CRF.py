import pandas as pd
import numpy as np
import pickle
import sys
import nltk
import sklearn
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_precision_score
from sklearn_crfsuite.metrics import flat_classification_report


df_test = pd.read_csv('dataset/test.csv', encoding = "ISO-8859-1")


df_test = df_test.fillna(method = 'ffill')
labels = list(df_test['Tag'].unique())


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

#Displaying one full sentence for test
getter_test = sentence(df_test)
sentences_test = [" ".join([s[0] for s in sent_test]) for sent_test in getter_test.sentences]
sentences_test[0]

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


X_test = [sent2features(s) for s in sentences_test]
y_test = [sent2labels(s) for s in sentences_test]


# Load from file
pkl_filename = sys.argv[1] + ".pkl"
with open(pkl_filename, 'rb') as file:
    crf = pickle.load(file)


#Predicting on the test set.
y_pred = crf.predict(X_test)

f1_score = flat_f1_score(y_test, y_pred, average = 'weighted')
print(f1_score)

print(multilabel_confusion_matrix(sum(y_test, []), sum(y_pred, []), labels=labels))	

report = flat_classification_report(y_test, y_pred)
print(report)

i = np.random.randint(0,len(sentences_test)-1) # choose a random number between 0 and len(sentences_test)b
# print(p)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for ((w, r), original, pred) in zip(sentences_test[i], y_test[i], y_pred[i]):
    if w != 0:
        print("{:15}: {:5} {}".format(w,original, pred))
