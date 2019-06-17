import pandas as pd
from googletrans import Translator
import malaya

#Initialize malaya function.
detLang = malaya.multinomial_detect_languages()
malays = malaya.load_malay_dictionary()
norm = malaya.fuzzy_normalizer(malays)

tweet = pd.read_csv("cleanEnglish.csv",sep = ";", encoding = "latin1")

#Normalise any Malay words in the data
temp = []
for i,row in tweet.iterrows():
    temp = []
    for k,word in enumerate(row['text'].split()):
        if detLang.predict(word) != 'ENGLISH':
            word = norm.normalize(word)
        temp.append(word)
    tweet.at[i,"normalise"] = " ".join(temp)

#Translate any Malay word into English
ts = Translator()
temp = []
for i,row in tweet.iterrows():
    temp = []
    for k,word in enumerate(row['text'].split()):
        if word == "Pos" or word == "Laju":
            pass
        elif detLang.predict(word) == "MALAY":
            word = ts.translate(word).text
        temp.append(word)
    tweet.at[i,"translate"] = " ".join(temp)

for i,row in tweet.iterrows():
	for k,word in enumerate(row['text'].split()):

		if len(word) < 3:
			continue
		else:
			temp.append(word)
	tweet.at[i,"text"] = " ".join(temp)

tweet.to_csv("translated.csv",sep = ",",index = False)