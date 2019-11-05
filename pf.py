import tensorflow as tf
import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import codecs


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

from ipywidgets import interact_manual
from ipywidgets import widgets

import re
import string



# Hyperparams if GPU is available
if tf.test.is_gpu_available():
    BATCH_SIZE = 512  # Number of examples used in each iteration
    EPOCHS = 8  # Number of passes through entire dataset
    MAX_LEN = 75  # Max length of review (in words)
    EMBEDDING = 40  # Dimension of word embedding vector

    
# Hyperparams for CPU training
else:
    BATCH_SIZE = 1024
    EPOCHS = 300
    MAX_LEN = 1000
    EMBEDDING = 50

csv.register_dialect('myDialect',
delimiter = ',',
skipinitialspace=True)

data = []
with os.scandir('dataset/') as entries:
 for entry in entries:
  with open("dataset/" + entry.name, 'r', errors='ignore') as csvFile:
    reader = csv.reader(( x.replace('\0', '') for x in csvFile), dialect='myDialect')
    for row in reader:
     data.append(row)
    #  print(row)
      
data.pop(0)
# print(data)

data.pop()

sentences = {}
for sentence in data:
 pos = sentence[0]
 if pos in sentences.keys():
  sentences[pos].append(sentence)
 else:
  sentences[sentence[0]] = [sentence]
print("Number of sentences: ", len(sentences))


words = []
ind = 0
for sentence in data:
#  print(ind)
#  print(sentence)
 ind = ind + 1
 word = sentence[1]
 if word not in words:
  words.append(word)
n_words = len(words)
print("Number of words: ", n_words)
# print(words)



# tags = ['FECHAS', 'CORREO_ELECTRONICO', 'PAIS', 'TERRITORIO', 'CALLE', 'NOMBRE_PERSONAL_SANITARIO', 'NOMBRE_PERSONAL_SANITARIO',
#  'EDAD_SUJETO_ASISTENCIA', 'ID_TITULACION_PERSONAL_SANITARIO', 'SEXO_SUJETO_ASISTENCIA', 'ID_ASEGURAMIENTO', 'ID_SUJETO_ASISTENCIA',
#  'NOMBRE_SUJETO_ASISTENCIA', 'FAMILIARES_SUJETO_ASISTENCIA', 'PROFESION', 'CENTRO_DE_SALUD', 'HOSPITAL', 'INSTITUCION', 'ID_EMPLEO_PERSONAL_SANITARIO',
#   'IDENTIF_VEHICULOS_NRSERIE_PLACA', 'IDENTIF_DISPOSITIVOS_NRSERIE', 'NUMERO_TELEFONO', 'NUMERO_FAX', 'ID_CONTACTO_ASISTENCIAL', 'NUMERO_BENEF_PLAN_SALUD',
#   'URL_WEB', 'DIREC_PROT_INTERNET', 'IDENTIF_BIOMETRICOS', 'OTRO_NUMERO_IDENTIF', 'OTROS_SUJETO_ASISTENCIA', '']

tags = ['E-CE', 'I-CE', 'B-CE', 'TERR', 'E-CALLE', 'I-CALLE', 'B-CALLE', 'B-ISA', 'E-ISA', 'I-ESA', 'PROF', 'B-PROF', 'I-PROF', 'E-PROF', 'NPS', 'B-INSTITUCION', 'I-INSTITUCION', 'E-INSTITUCION',
 'E-HOS', 'I-HOS', 'B-HOS', 'NSA', 'E-NPS', 'B-IA', 'I-IA', 'E-IA', 'IA', 'B-FSA', 'I-FSA', 'E-FSA', 'I-ISA', 'F', 'B-PAIS', 'I-PAIS', 'E-PAIS',
 'I-NPS', 'B-NPS', 'B-NT', 'I-NT', 'E-NT', 'E-ITPS', 'I-ITPS', 'B-TERR', 'I-TERR', 'E-TERR', 'ISA', 'NT', 'B-CS', 'I-CS', 'E-CS', 'B-OSA', 'I-OSA', 'E-OSA',
  'B-ITPS', 'E-F', 'I-F', 'B-F', 'SSA', 'ESA', 'B-EDA', 'I-EDA', 'E-EDA', 'FSA', 'B-HOSP', 'I-HOSP', 'E-HOSP', 'OSA', 'INSTITUCION', 'HOSP',
  'ISA', 'B-ESA', 'E-ESA', 'I-IEPS', 'B-IEPS', 'E-IEPS', 'I-NSA', 'B-NSA', 'E-NSA', 'PAIS', 'ITPS', '', 'FECHA', 'ICA']
print("Tags:", tags)
n_tags = len(tags)
print("Number of Labels: ", n_tags)



# Vocabulary Key:word -> Value:token_index
# The first 2 entries are reserved for PAD and UNK
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1 # Unknown words
word2idx["PAD"] = 0 # Padding

# Vocabulary Key:token_index -> Value:word
idx2word = {i: w for w, i in word2idx.items()}

# Vocabulary Key:Label/Tag -> Value:tag_index
# The first entry is reserved for PAD
tag2idx = {t: i+1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0

# Vocabulary Key:tag_index -> Value:Label/Tag
idx2tag = {i: w for w, i in tag2idx.items()}

print("The word Miguel is identified by the index: {}".format(word2idx["hematuria"]))
print("The labels B-geo(which defines Geopraphical Enitities) is identified by the index: {}".format(tag2idx["I-NSA"]))


from keras.preprocessing.sequence import pad_sequences
# Convert each sentence from list of Token to list of word_index
X = [[word2idx[w[1]] for w in s] for s in sentences.values()]
# Padding each sentence to have the same lenght
X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

# Convert Tag/Label to tag_index
y = [[tag2idx[w[3]] for w in s] for s in sentences.values()]
# Padding each sentence to have the same lenght
y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])

from keras.utils import to_categorical
# One-Hot encode
y = [to_categorical(i, num_classes=n_tags+1) for i in y]  # n_tags+1(PAD)

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape

# print('Raw Sample: ', ' '.join([w[1] for w in sentences['1']]))
# print('Raw Label: ', ' '.join([w[3] for w in sentences['1']]))
print('After processing, sample:', X[0])
print('After processing, labels:', y[0])



# Model definition
input = Input(shape=(MAX_LEN,))
model = Embedding(input_dim=n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                  input_length=MAX_LEN, mask_zero=True)(input)  # default: 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags+1)  # CRF layer, n_tags+1(PAD)
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.1, verbose=2)

# Eval
pred_cat = model.predict(X_te)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_te, -1)

from sklearn_crfsuite.metrics import flat_classification_report

# Convert the index to tag
pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 

report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)


i = np.random.randint(0,X_te.shape[0]) # choose a random number between 0 and len(X_te)
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_te[i], -1)

print("Sample number {} of {} (Test Set)".format(i, X_te.shape[0]))
# Visualization
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_te[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-2], idx2tag[t], idx2tag[pred]))


# Custom Tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
    
def get_prediction(sentence):
    test_sentence = tokenize(sentence) # Tokenization
    # Preprocessing
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
                            padding="post", value=word2idx["PAD"], maxlen=MAX_LEN)
    # Evaluation
    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    # Visualization
    print("{:15}||{}".format("Word", "Prediction"))
    print(30 * "=")
    for w, pred in zip(test_sentence, p[0]):
        print("{:15}: {:5}".format(w, idx2tag[pred]))

mysentence = "Datos del paciente. Nombre:  Ignacio. Apellidos: Rico Pedroza. NHC: 5467980. Domicilio: Av. Beniarda, 13. Localidad/ Provincia: Valencia. CP: 46271. Datos asistenciales. Fecha de nacimiento: 11/02/1970. País: España. Edad: 46 años Sexo: H. Fecha de Ingreso: 28/05/2016. Médico:  Ignacio Rubio Tortosa Servicio  NºCol: 46 28 52938. Informe clínico del paciente: Paciente de 46 años que consultó por dolor a nivel de hipogastrio, eyaculación dolorosa, hemospermia y sensación de peso a nivel testicular atribuido hasta entonces a varicocele derecho ya conocido desde hacía un año. Entre sus antecedentes personales destacaba un episodio de prostatitis aguda un año antes de la consulta. A la exploración física el paciente presentaba buen estado general, varicocele derecho y no se palpaban masas a nivel de ambos testículos. vasculares capsulares adyacentes y músculo liso; moderada atípia y escasas mitosis. Tras 10 años de controles evolutivos, el paciente se encuentra asintomático, no se han objetivado metástasis y los marcadores tumorales han permanecido negativos. Remitido por: Dr.Ignacio Rubio Tortosa Servicio de Urología Hospital Dr. Peset Avda. Gaspar Aguilar, 90 46017 Valencia. (España) e-mail: nachorutor@hotmail.com "
get_prediction(mysentence)
