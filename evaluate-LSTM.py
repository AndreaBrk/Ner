import pandas as pd
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import load_model  
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from sklearn_crfsuite.metrics import flat_classification_report


from sklearn.metrics import f1_score
import pickle


#Reading the csv file
df = pd.read_csv('dataset/dataset.csv', encoding = "ISO-8859-1")
#Reading the csv file test
df_test = pd.read_csv('dataset/test.csv', encoding = "ISO-8859-1")

df_test.isnull().sum()
df = df.fillna(method = 'ffill')
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
max_len = 1200 

#Displaying one full sentence test
getter_test = sentence(df_test)
sentences_test = [" ".join([s[0] for s in sent]) for sent in getter_test.sentences]
sentences_test[0]


sentences_test = getter_test.sentences

with open('word_to_index.pickle', 'rb') as f:
    word_to_index_test = pickle.load(f)


with open('tag_to_index.pickle', 'rb') as f:
    tag_to_index_test = pickle.load(f)

print(tag_to_index_test)

idx2tag = {i: w for w, i in tag_to_index_test.items()}

X_test = [[word_to_index_test[w[0]] for w in s] for s in sentences_test]
X_test = pad_sequences(maxlen = max_len, sequences = X_test, padding = "post", value = word_to_index_test["PAD"])

y_test = [[tag_to_index_test[w[2]] for w in s] for s in sentences_test]
y_test = pad_sequences(maxlen = max_len, sequences = y_test, padding = "post", value = tag_to_index_test["PAD"])

num_tag_test = df_test['Tag'].nunique()
y_test = [to_categorical(i, num_classes = num_tag_test + 1) for i in y_test]


num_tags = df_test['Tag'].nunique()

crf = CRF(num_tags+1)  # CRF layer

filename = 'finalized_model_index.sav'


model= load_model(filename,custom_objects={'CRF':CRF, 
                                                  'crf_loss':crf_loss, 
                                                  'crf_viterbi_accuracy':crf_viterbi_accuracy})


# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, -1)

# Convert the index to tag
y_pred = [[idx2tag[i] for i in row] for row in y_pred]
y_test_true = [[idx2tag[i] for i in row] for row in y_test_true]

# print("F1-score is : {:.1%}".format(f1_score(y_test_true, y_pred)))
report = flat_classification_report(y_pred=y_pred, y_true=y_test_true)
print(report)



#Getting unique words and labels from data
words = list(df['Word'].unique())

#Getting unique words and labels from data
words_test = list(df_test['Word'].unique())
# tags_test = list(df_test['Tag'].unique())

words = words + words_test
# print(words)


# At every execution model picks some random test sample from test set.
i = np.random.randint(0,X_test.shape[0]) # choose a random number between 0 and len(X_te)b
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test[i], -1)

# print("Sample number {} of {} (Test Set)".format(i, X_test.shape[0]))
# # Visualization
# print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
# print(30 * "=")
# for w, t, pred in zip(X_test[i], true, p[0]):
#     if w != 0:
#         print("{:15}: {:5} {}".format(words[w-2], idx2tag[t], idx2tag[pred]))
