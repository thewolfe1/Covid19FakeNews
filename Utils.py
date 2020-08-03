import joblib
import twint
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from nltk.corpus import stopwords
import pickle

def load_data():

    c = twint.Config()
    c.Limit = 5
    c.Store_csv = True
    c.Output = "tweets.csv"
    c.Lang = "en"
    c.Search = "covid-19"

    twint.run.Search(c)

def clear_csv():
    f = open("tweets.csv", "w")
    f.truncate()
    f.close()

def read_data():
    df = pd.read_csv("tweets.csv")
    return df['tweet']

def num_to_bool(result):
    if result==0:
        return 'Fake'
    return 'True'

def clean_tweet(s):

    stemmer = WordNetLemmatizer()

    # Remove all the special characters
    sentence = re.sub(r'\W', ' ', str(s))

    # remove all single characters
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

    # Remove single characters from the start
    sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)

    # Substituting multiple spaces with single space
    sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

    # Removing prefixed 'b'
    sentence = re.sub(r'^b\s+', '', sentence)

    # Converting to Lowercase
    sentence = sentence.lower()

    # Lemmatization
    sentence = sentence.split()

    sentence = [stemmer.lemmatize(word) for word in sentence]
    sentence = ' '.join(sentence)

    return sentence

def preprocess(tweets,max_features=1500):
    # Tokenizing the text - converting the words, letters into counts or numbers.
    # We dont need to explicitly remove the punctuations. we have an inbuilt option in Tokenizer for this purpose
    tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

    tokenizer.fit_on_texts(texts=tweets)
    s = tokenizer.texts_to_sequences(texts=tweets)

    # now applying padding to make them even shaped.
    return pad_sequences(sequences=s, maxlen=max_features, padding='pre')

def predict_lstm(tweets,max_features=1500):

    x=preprocess(tweets,max_features)
    model = load_model('lstm.h5')
    return model.predict_classes(x)

def predict_gru(tweets,max_features=1500):

    x = preprocess(tweets, max_features)
    model = load_model('gru.h5')
    return model.predict_classes(x)

#print(predict_lstm(read_data()).tolist())
def clean_str(s):
    # clean tweets
    stemmer = WordNetLemmatizer()

    # Remove all the special characters
    sentence = re.sub(r'\W', ' ', str(s))

    # remove all single characters
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

    # Remove single characters from the start
    sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)

    # Substituting multiple spaces with single space
    sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

    # Removing prefixed 'b'
    sentence = re.sub(r'^b\s+', '', sentence)

    # Converting to Lowercase
    sentence = sentence.lower()

    # Lemmatization
    sentence = sentence.split()

    sentence = [stemmer.lemmatize(word) for word in sentence]
    sentence = ' '.join(sentence)

    return sentence

def preprocess_alg(tweets):
    documents = []
    for sen in range(0, len(tweets)):
        documents.append(clean_str(tweets[sen]))
    # bag of words
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    tfidfconverter = pickle.load(open("tfidfconverter.pickle", "rb"))
    X = vectorizer.transform(documents).toarray()
    # tf-idf
    return tfidfconverter.transform(X).toarray()

def predict_svc(tweets):

    # load the model from disk
    model = joblib.load('svc.sav')
    return model.predict(preprocess_alg(tweets))

def predict_LR(tweets):

    # load the model from disk
    model = joblib.load('LR.sav')
    return model.predict(preprocess_alg(tweets))

def predict_NB(tweets):

    # load the model from disk
    model = joblib.load('NB.sav')
    return model.predict(preprocess_alg(tweets))

def predict_MNB(tweets):

    # load the model from disk
    model = joblib.load('MNB.sav')
    return model.predict(preprocess_alg(tweets))

def predict_GNB(tweets):

    # load the model from disk
    model = joblib.load('GNB.sav')
    return model.predict(preprocess_alg(tweets))

def predict_RF(tweets):

    # load the model from disk
    model = joblib.load('RF.sav')
    return model.predict(preprocess_alg(tweets))

#load_data()
#print(predict_RF(read_data()).tolist())