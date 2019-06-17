import pandas as pd
import re
from langdetect import detect

#Read CSV
tweet2 = pd.read_csv("tweet1.csv",encoding = "latin1",sep=";")
tweet3 = pd.read_csv("tweet2.csv",encoding = "latin1",sep=";")
#Combine the csv
tweet = pd.concat([tweet2,tweet3])

#Detect English Tweets
tweet1 = pd.DataFrame()
for i,row in tweet.iterrows():
    temp = detect(row['text'])
    if temp == 'en':
        tweet1.at[i,'text'] = row['text']
print("Finished Filtering !!!")

#Reset Index
tweet1= tweet1.reset_index(drop=True)

#Drop Unnamed Column
tweet1 = tweet1.drop(tweet1.columns[tweet1.columns.str.contains('unnamed',case = False)],axis = 1)
#Drop duplicate
tweet1 = tweet1.drop_duplicates(keep='first')
#Reset index of the df
tweet1= tweet1.reset_index(drop=True)
#Delete https
for i, row in tweet1.iterrows():
    temp = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", row['text'])
    tweet1.at[i,'text'] = temp
#Inverse to lowercase
tweet1['text'] = tweet1['text'].str.lower()
#Delete unwanted tweets
unW = ["i'm at","jual","sell","promosi","mayor","restock","stock","rm"]
for item in unW:
    tweet1 = tweet1[tweet1.text.str.contains(item) == False]
    
#Delete Tags and Hashtags words
for i, row in tweet1.iterrows():
    tweet1.at[i,'text'] = " ".join(filter(lambda x:x[0]!='@', row['text'].split()))
tweet1['text'].replace('#', '', regex=True)

#Delete symbols and punctuation
for i, row in tweet1.iterrows():
    temp = re.sub(r'[^\w]', ' ', row['text'])
    tweet1.at[i,'text'] = temp
    
#Delete text with less than 4 words
for i,row in tweet1.iterrows():
    if len(row['text'].split()) < 4:
        tweet1 = tweet1.drop(i)

#Delete multiple white space
tweet1.text = tweet1.text.replace('\s+', ' ', regex=True)    
    
#Replace poslaju and pos laju to nouns 
tweet1['text'].replace({'poslaju':'Pos Laju'},inplace = True, regex = True)
tweet1['text'].replace({'pos laju':'Pos Laju'},inplace = True, regex = True)

tweet1 = tweet1.reset_index(drop=True)

#Save Cleaned Data
tweet1.to_csv("cleanEnglish1.csv", sep=',', encoding='utf-8',index = False)