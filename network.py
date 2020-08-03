import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM

data = pd.read_csv('out - out.csv')
data_copy = data.copy()


data_copy = data_copy.set_index('id', drop = True)
print(data_copy)

# checking for missing values
print(data_copy.isnull().sum())


length = []
[length.append(len(str(text))) for text in data_copy['text']]
data_copy['length'] = length
data_copy.head()

max_features = 1500

sen=["covid-19 is bad is what i think","breakingnews covid 19 is back","trump said covid wow"]
# Tokenizing the text - converting the words, letters into counts or numbers.
# We dont need to explicitly remove the punctuations. we have an inbuilt option in Tokenizer for this purpose
tokenizer = Tokenizer(num_words = max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower = True, split = ' ')
tokenizer.fit_on_texts(texts = data_copy['text'])
X = tokenizer.texts_to_sequences(texts = data_copy['text'])

tokenizer.fit_on_texts(texts = sen)
s= tokenizer.texts_to_sequences(texts = sen)
print(s)

# now applying padding to make them even shaped.
X = pad_sequences(sequences = X, maxlen = max_features, padding = 'pre')
s= pad_sequences(sequences = s, maxlen = max_features, padding = 'pre')

print(X.shape)
y=[]
#y = data_copy['label'].values
for label in data_copy['label']:
    if label is 'T':
        y.append(1)
    else:
        y.append(0)

y= np.array(y)
print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# LSTM Neural Network
lstm_model = Sequential(name = 'lstm_nn_model')
lstm_model.add(layer = Embedding(input_dim = max_features, output_dim = 120, name = '1st_layer'))
lstm_model.add(layer = LSTM(units = 120, dropout = 0.2, recurrent_dropout = 0.2, name = '2nd_layer'))
lstm_model.add(layer = Dropout(rate = 0.5, name = '3rd_layer'))
lstm_model.add(layer = Dense(units = 120,  activation = 'relu', name = '4th_layer'))
lstm_model.add(layer = Dropout(rate = 0.5, name = '5th_layer'))
lstm_model.add(layer = Dense(units = len(set(y)),  activation = 'sigmoid', name = 'output_layer'))
# compiling the model
lstm_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy','mse'])

lstm_model_fit = lstm_model.fit(X_train, y_train, epochs = 1)
scores = lstm_model.evaluate(X_test, y_test, verbose=0)
y_pred = lstm_model.predict(X_test)
print(scores)
print(classification_report(y_test,y_pred.argmax(axis=1)))

lstm_model.save('lstm.h5')
lstm_model.save_weights('lstm_weights.h5')


# GRU neural Network
gru_model = Sequential(name = 'gru_nn_model')
gru_model.add(layer = Embedding(input_dim = max_features, output_dim = 120, name = '1st_layer'))
gru_model.add(layer = GRU(units = 120, dropout = 0.2,
                          recurrent_dropout = 0.2, recurrent_activation = 'relu',
                          activation = 'relu', name = '2nd_layer'))
gru_model.add(layer = Dropout(rate = 0.4, name = '3rd_layer'))
gru_model.add(layer = Dense(units = 120, activation = 'relu', name = '4th_layer'))
gru_model.add(layer = Dropout(rate = 0.2, name = '5th_layer'))
gru_model.add(layer = Dense(units = len(set(y)), activation = 'softmax', name = 'output_layer'))
# compiling the model
gru_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy','mse'])

gru_model.summary()

gru_model_fit = gru_model.fit(X_train, y_train, epochs = 1)
scores = gru_model.evaluate(X_test, y_test, verbose=0)
print(scores)
y_pred = gru_model.predict(X_test)
lstm_model.save('gru.h5')
lstm_model.save_weights('gru_weights.h5')

lstm_prediction = lstm_model.predict_classes(s)
print(lstm_prediction)

print(classification_report(y_test,y_pred.argmax(axis=1)))