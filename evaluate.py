import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# from sklearn.externals import joblib

from keras.models import load_model  
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from sklearn_crfsuite.metrics import flat_classification_report


from sklearn.metrics import f1_score




#Reading the csv file test
df_test = pd.read_csv('dataset/test.csv', encoding = "ISO-8859-1")

df_test.isnull().sum()
df_test = df_test.fillna(method = 'ffill')

# This is a class te get sentence. The each sentence will be list of tuples with its tag and pos.
class sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s : [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                       s['POS'].values.tolist(),
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

# Maximum length of review
max_len = 500 

#Displaying one full sentence test
getter_test = sentence(df_test)
sentences_test = [" ".join([s[0] for s in sent]) for sent in getter_test.sentences]
sentences_test[0]


#sentence with its pos and tag in test
sent_test = getter_test.get_text()
print(sent_test)

sentences_test = getter_test.sentences

#Getting unique words and labels from data
words_test = list(df_test['Word'].unique())
tags_test = list(df_test['Tag'].unique())

# Dictionary word:index pair
# word is key and its value is corresponding index
word_to_index_test = {w : i + 2 for i, w in enumerate(words_test)}
word_to_index_test["UNK"] = 1
word_to_index_test["PAD"] = 0

# Dictionary lable:index pair
# label is key and value is index.
tag_to_index_test = {t : i + 1 for i, t in enumerate(tags_test)}
tag_to_index_test["PAD"] = 0

idx2word_test = {i: w for w, i in word_to_index_test.items()}
idx2tag_test = {i: w for w, i in tag_to_index_test.items()}

X_test = [[word_to_index_test[w[0]] for w in s] for s in sentences_test]
X_test = pad_sequences(maxlen = max_len, sequences = X_test, padding = "post", value = word_to_index_test["PAD"])

y_test = [[tag_to_index_test[w[2]] for w in s] for s in sentences_test]
y_test = pad_sequences(maxlen = max_len, sequences = y_test, padding = "post", value = tag_to_index_test["PAD"])

num_tag_test = df_test['Tag'].nunique()
y_test = [to_categorical(i, num_classes = num_tag_test + 1) for i in y_test]





num_tags = df_test['Tag'].nunique()

crf = CRF(num_tags+1)  # CRF layer

filename = 'finalized_model.sav'

# model = joblib.load(filename)

model= load_model(filename,custom_objects={'CRF':CRF, 
                                                  'crf_loss':crf_loss, 
                                                  'crf_viterbi_accuracy':crf_viterbi_accuracy})


# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, -1)
print ("----------------")

# # Convert the index to tag
# y_pred = [[idx2tag[i] for i in row] for row in y_pred]
# y_test_true = [[idx2tag[i] for i in row] for row in y_test_true]

# Convert the index to tag
y_pred = [[idx2tag_test[i] for i in row] for row in y_pred]
y_test_true = [[idx2tag_test[i] for i in row] for row in y_test_true]

print(y_pred[0])
print ("----------------")
print ("----------------")
print(y_test_true[0])

# print("F1-score is : {:.1%}".format(f1_score(y_test_true, y_pred)))
# report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
report = flat_classification_report(y_test_true, y_pred)
print(report)

# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================


# df_test = pd.read_csv('dataset/test.csv', encoding = "ISO-8859-1")

# df_test.isnull().sum()
# df_test = df_test.fillna(method = 'ffill')

# # This is a class te get sentence. The each sentence will be list of tuples with its tag and pos.
# class sentence(object):
#     def __init__(self, df):
#         self.n_sent = 1
#         self.df = df
#         self.empty = False
#         agg = lambda s : [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
#                                                        s['POS'].values.tolist(),
#                                                        s['Tag'].values.tolist())]
#         self.grouped = self.df.groupby("Sentence #").apply(agg)
#         self.sentences = [s for s in self.grouped]
        
#     def get_text(self):
#         try:
#             s = self.grouped['Sentence: {}'.format(self.n_sent)]
#             self.n_sent +=1
#             return s
#         except:
#             return None




# #Displaying one full sentence for test
# getter_test = sentence(df_test)
# sentences_test = [" ".join([s[0] for s in sent_test]) for sent_test in getter_test.sentences]
# sentences_test[0]

# #sentence with its pos and tag for test.
# sent_test = getter_test.get_text()
# sentences_test = getter_test.sentences

# def word2features(sent, i):
#     word = sent[i][0]
#     postag = sent[i][1]

#     features = {
#         'bias': 1.0,
#         'word.lower()': word.lower(),
#         'word[-3:]': word[-3:],
#         'word[-2:]': word[-2:],
#         'word.isupper()': word.isupper(),
#         'word.istitle()': word.istitle(),
#         'word.isdigit()': word.isdigit(),
#         'postag': postag,
#         'postag[:2]': postag[:2],
#     }
#     if i > 0:
#         word1 = sent[i-1][0]
#         postag1 = sent[i-1][1]
#         features.update({
#             '-1:word.lower()': word1.lower(),
#             '-1:word.istitle()': word1.istitle(),
#             '-1:word.isupper()': word1.isupper(),
#             '-1:postag': postag1,
#             '-1:postag[:2]': postag1[:2],
#         })
#     else:
#         features['BOS'] = True

#     if i < len(sent)-1:
#         word1 = sent[i+1][0]
#         postag1 = sent[i+1][1]
#         features.update({
#             '+1:word.lower()': word1.lower(),
#             '+1:word.istitle()': word1.istitle(),
#             '+1:word.isupper()': word1.isupper(),
#             '+1:postag': postag1,
#             '+1:postag[:2]': postag1[:2],
#         })
#     else:
#         features['EOS'] = True
#     return features

# def sent2features(sent):
#     return [word2features(sent, i) for i in range(len(sent))]

# def sent2labels(sent):
#     return [label for token, postag, label in sent]

# def sent2tokens(sent):
#     return [token for token, postag, label in sent]


# X_test = [sent2features(s) for s in sentences_test]
# y_test = [sent2labels(s) for s in sentences_test]

# num_tags = df_test['Tag'].nunique()

# crf = CRF(num_tags+1)  # CRF layer

# filename = 'finalized_model.sav'

# # model = joblib.load(filename)

# model= load_model(filename,custom_objects={'CRF':CRF, 
#                                                   'crf_loss':crf_loss, 
#                                                   'crf_viterbi_accuracy':crf_viterbi_accuracy})

# y_pred = model.predict(X_test)
# y_pred = np.argmax(y_pred, axis=-1)
# # y_test_true = np.argmax(y_test, -1)

# report = flat_classification_report(y_test, y_pred)
