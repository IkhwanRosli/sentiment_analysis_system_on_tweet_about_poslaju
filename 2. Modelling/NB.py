import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data.csv", sep = ";")

vector = TfidfVectorizer(use_idf = True, lowercase = True, strip_accents = 'ascii')

y = df.label
x = vector.fit_transform(df.text)

clf = naive_bayes.MultinomialNB()
clf.fit(x_train, y_train)

print("THE Accuracy of the Model is {}%".format(roc_auc_score(y_test, clf.predict_proba(x_test)[:,1]))*100.0)