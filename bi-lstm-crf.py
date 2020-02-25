import pandas as pd
import numpy as np
import pickle
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import f1_score
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report



#Reading the csv file
df = pd.read_csv('dataset/dataset.csv', encoding = "ISO-8859-1")

#Reading the csv file test
df_test = pd.read_csv('dataset/test.csv', encoding = "ISO-8859-1")

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

#Displaying one full sentence test
getter_test = sentence(df_test)
sentences_test = [" ".join([s[0] for s in sent]) for sent in getter_test.sentences]
sentences_test[0]


#sentence with its pos and tag.
sent = getter.get_text()
sent_test = getter_test.get_text()
sentences = getter.sentences
sentences_test = getter_test.sentences


# Number of data points passed in each iteration
batch_size = 512 
# Passes through entire dataset
epochs = int(sys.argv[1])
# Maximum length of reports
max_len = int(sys.argv[2]) 
# Dimension of embedding vector
embedding = 50

print(epochs)
print(max_len)
#Getting unique words and labels from data
words = list(df['Word'].unique())
tags = list(df['Tag'].unique())

#Getting unique words and labels from data
words_test = list(df_test['Word'].unique())

words = words + words_test

# Dictionary word:index pair
# word is key and its value is corresponding index
word_to_index = {w : i + 2 for i, w in enumerate(words)}
# word_to_index["UNK"] = 1
word_to_index["O"] = 0

# Dictionary lable:index pair
# label is key and value is index.
tag_to_index = {t : i + 1 for i, t in enumerate(tags)}
tag_to_index["O"] = 0

idx2word = {i: w for w, i in word_to_index.items()}
idx2tag = {i: w for w, i in tag_to_index.items()}


# Converting each sentence into list of index from list of tokens
X = [[word_to_index[w[0]] for w in s] for s in sentences]
X_test = [[word_to_index[w[0]] for w in s] for s in sentences_test]


# Padding each sequence to have same length  of each word
X = pad_sequences(maxlen = max_len, sequences = X, padding = "post")
X_test = pad_sequences(maxlen = max_len, sequences = X_test, padding = "post")


# Convert label to index
y = [[tag_to_index[w[1]] for w in s] for s in sentences]
y_test = [[tag_to_index[w[1]] for w in s] for s in sentences_test]


# padding
y = pad_sequences(maxlen = max_len, sequences = y, padding = "post")
y_test = pad_sequences(maxlen = max_len, sequences = y_test, padding = "post")

num_tag = df['Tag'].nunique()

# One hot encoded labels
y = [to_categorical(i, num_classes = num_tag + 1) for i in y]
y_test = [to_categorical(i, num_classes = num_tag + 1) for i in y_test]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

X_train = X
y_train = y

print("Size of training input data : ", X_train.shape)
print("Size of training output data : ", np.array(y_train).shape)
print("Size of testing input data : ", X_test.shape)
print("Size of testing output data : ", np.array(y_test).shape)


# # Let's check the first sentence before and after processing.
# print('*****Before Processing first sentence : *****\n', ' '.join([w[0] for w in sentences[0]]))
# print('*****After Processing first sentence : *****\n ', X[0])

# # First label before and after processing.
# print('*****Before Processing first sentence : *****\n', ' '.join([w[1] for w in sentences[0]]))
# print('*****After Processing first sentence : *****\n ', y[0])

num_tags = df['Tag'].nunique()
# Model architecture
input = Input(shape = (max_len,))
model = Embedding(input_dim = len(words) + 2, output_dim = embedding, input_length = max_len, mask_zero = True)(input)
model = Bidirectional(LSTM(units = 50, return_sequences=True, recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)
crf = CRF(num_tags+1)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])


model.summary()

checkpointer = ModelCheckpoint(filepath = 'model.h5',
                       verbose = 0,
                       mode = 'auto',
                       save_best_only = True,
                       monitor='val_loss')

history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=epochs,
                    validation_split=0.1, callbacks=[checkpointer])

# plotting
acc = history.history['crf_viterbi_accuracy']
val_acc = history.history['val_crf_viterbi_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(figsize = (8, 8))
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure(figsize = (8, 8))
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


with open('word_to_index_O_500.pickle', 'wb') as file:
    pickle.dump(word_to_index, file)


with open('tag_to_index_O_500.pickle', 'wb') as file:
    pickle.dump(tag_to_index, file)



path = sys.argv[3] + '.sav'
model.save(path)




# Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)
y_test_true = np.argmax(y_test, -1)

# Convert the index to tag
y_pred = [[idx2tag[i] for i in row] for row in y_pred]
y_test_true = [[idx2tag[i] for i in row] for row in y_test_true]

print("F1-score is : {:.1%}".format(f1_score(y_test_true, y_pred)))
report = flat_classification_report(y_pred=y_pred, y_true=y_test_true, labels=tags)
print(report)
