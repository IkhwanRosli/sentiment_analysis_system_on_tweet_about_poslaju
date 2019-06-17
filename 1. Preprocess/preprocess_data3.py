import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.stem import PorterStemmer

tweet = pd.read_csv("translated.csv",sep =";",encoding="latin1")

lemma = nltk.wordnet.WordNetLemmatizer()

#Stemming, Lemmalization and Deleting the Stopwords.
wn = nltk.wordnet.WordNetLemmatizer()
lc = nltk.stem.SnowballStemmer('english')
#ps = PorterStemmer()
sw = set(stopwords.words('english'))
hasStop = tweet['text'].tolist()
noStop = []
for item in hasStop:
    filtered = []
    wt = word_tokenize(item)
    for wo in wt:
        if wo == "not":
            filtered.append(wo)
        elif not wo in sw:
            filtered.append(wo)
    filtered = [wn.lemmatize(w) for w in filtered]
    filtered = [lc.stem(w) for w in filtered]
    noStop.append(' '.join(filtered))
temp = pd.Series(noStop)
tweet['text'] = temp.values

tweet.to_csv("data.csv",sep = ",",index = False)