import pandas as pd
from itertools import chain


import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer  
from sklearn.metrics import multilabel_confusion_matrix


from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_precision_score
from sklearn_crfsuite.metrics import flat_classification_report



from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


#Reading the csv file
df = pd.read_csv('dataset/dataset.csv', encoding = "ISO-8859-1")
df_test = pd.read_csv('dataset/test.csv', encoding = "ISO-8859-1")


#Display first 10 rows
df.head(10)

df.describe()


#Displaying the unique Tags
labels = list(df['Tag'].unique())
print(labels)
# labels = list(crf.classes_)
labels.remove('PAD')
print(labels)

#Checking null values, if any.
df.isnull().sum()
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

#Displaying one full sentence
getter = sentence(df)
sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sentences[0]

#Displaying one full sentence for test
getter_test = sentence(df_test)
sentences_test = [" ".join([s[0] for s in sent_test]) for sent_test in getter_test.sentences]
sentences_test[0]


#sentence with its pos and tag.
sent = getter.get_text()
# print("sent")
# print(sent)

#sentence with its pos and tag for test.
sent_test = getter_test.get_text()
# print("sent_test")
# print(sent_test)

sentences = getter.sentences
sentences_test = getter_test.sentences

# print("sentences")
# print(sentences)
# print(sentences_test)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

X_test = [sent2features(s) for s in sentences_test]
y_test = [sent2labels(s) for s in sentences_test]

# divide el set en test y train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train = X
y_train = y

# crf = CRF(algorithm = 'lbfgs',
#          c1 = 0.4,
#          c2 = 0.4,
#          max_iterations = 100,
#          all_possible_transitions = False)



crf = CRF(algorithm = 'l2sgd',
         c2 = 1,
         max_iterations = 1000,
         all_possible_transitions = False)


# crf = CRF(
#     algorithm='lbfgs',
#     max_iterations=100,
#     all_possible_transitions=True
# )
# params_space = {
#     'c1': scipy.stats.expon(scale=0.5),
#     'c2': scipy.stats.expon(scale=0.05),
# }

# # use the same metric for evaluation
# f1_scorer = make_scorer(metrics.flat_f1_score,
#                         average='weighted', labels=labels)

# # search
# rs = RandomizedSearchCV(crf, params_space,
#                         cv=3,
#                         refit=True,
#                         verbose=1,
#                         n_jobs=-1,
#                         n_iter=50,
#                         scoring=f1_scorer)

                        
crf.fit(X_train, y_train)
# rs.fit(X_train, y_train)


# crf = rs.best_estimator_
# print('best params:', rs.best_params_)
# print('best CV score:', rs.best_score_)
# print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

#Predicting on the test set.
y_pred = crf.predict(X_test)

f1_score = flat_f1_score(y_test, y_pred, average = 'weighted')
print(f1_score)

accuracy = flat_precision_score(y_test, y_pred, average = 'weighted')
print(accuracy)



report = flat_classification_report(y_test, y_pred)
print(report)

# y_test = MultiLabelBinarizer().fit_transform(y_test)
# y_pred = MultiLabelBinarizer().fit_transform(y_pred)  
# print (y_test)
# print(classification_report(y_test, y_pred))
# print(classification_report(y_test, y_pred,labels = labels))
print(multilabel_confusion_matrix(sum(y_test, []), sum(y_pred, []), labels=labels))

# print(confusion_matrix(
#     y_test.values.argmax(axis=1), predictions.argmax(axis=1)))

# sklearn.metrics.f1_score(y_test, y_pred, average='micro')
