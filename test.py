import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.svm import SVC
import pickle
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences



Tweets_df = pd.read_csv('out - out.csv',encoding="utf8")

df = pd.DataFrame({'id': range(len(Tweets_df['text'])),'username':Tweets_df['username'], 'text': Tweets_df['text'],'label':Tweets_df['label']}).set_index('id')

df_try=df
x=list(df_try['text'])

y=[]

for label in list(df_try['label']):
    if label is 'T':
        y.append(1)
    else:
        y.append(0)

y= np.array(y)



documents = []

#clean tweets
stemmer = WordNetLemmatizer()

def clean_str(s):
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

for sen in range(0, len(x)):
    documents.append(clean_str(x[sen]))


#bag of words
vectorizer = CountVectorizer(max_features=1500, min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()


#tf-idf
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
pickle.dump(tfidfconverter, open("tfidfconverter.pickle", "wb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#a try to predict a single sentense



#models

#svc
clf = SVC()
clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
y_pred = clf.predict(X_test)


print('svc')
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
# save the model to disk
joblib.dump(clf, 'svc.sav')


#Logistic Regression
clf2 = LogisticRegression(random_state=0)
clf2.fit(X, y)
y_pred = clf2.predict(X_test)
print('Logistic Regression')
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
# save the model to disk
joblib.dump(clf2, 'LR.sav')


#Naive Bayes
clf3 = BernoulliNB()
clf3.fit(X, y)
y_pred = clf3.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
# save the model to disk
joblib.dump(clf3, 'NB.sav')

clf4 = MultinomialNB()
clf4.fit(X, y)
y_pred = clf4.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
# save the model to disk
joblib.dump(clf4, 'MNB.sav')

clf5 = GaussianNB()
clf5.fit(X, y)
y_pred = clf5.predict(X_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
# save the model to disk
joblib.dump(clf5, 'GNB.sav')


#random forest

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# save the model to disk
joblib.dump(classifier, 'RF.sav')

#evaluate
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

