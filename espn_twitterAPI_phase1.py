#!/usr/bin/env python
# coding: utf-8

# In[990]:


get_ipython().system('pip install snscrape')
#!pip install nltk


# In[991]:


import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


# In[992]:


query = "(from:espn)"
tweets = []
limit = 1000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])
# print(df)


# In[993]:


df.head(10)


# In[994]:


df.shape


# In[997]:


df.to_csv("espn_task1.csv", index=False)


# In[998]:


ndf=pd.read_csv("espn_task1.csv")
ndf


# # Data Cleaning

# In[999]:


# convert to lower case
ndf['lower_case']=ndf['Tweet'].apply(lambda x: x.lower())


# In[1000]:


ndf


# In[1001]:


def remove_emoji(string):
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F" 
                u"\U0001F300-\U0001F5FF"  
                u"\U0001F680-\U0001F6FF"  
                u"\U0001F1E0-\U0001F1FF"  
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


# In[1002]:


# remove emojis 
ndf['rm_emojis'] = ndf['lower_case'].apply(lambda x: remove_emoji(x))
ndf.head(5)


# In[1003]:


ndf['removing_urls'] = ndf['rm_emojis'].apply(lambda x: re.sub(r'http\S+',"", x))
ndf.head(5)


# In[1004]:


ndf['removing@s'] = ndf['removing_urls'].apply(lambda x: ' '. join([word for word in x.split() if not word.startswith("@")]))
ndf.head(5)


# In[1005]:


ndf['removing@s']


# In[1006]:


ndf['removing_'] = ndf['removing@s'].apply(lambda x: ' '. join([word for word in x.split() if not word.startswith("_")]))
ndf.head(5)


# In[1007]:


stop_words = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]


# In[1008]:


def remove_stopwords(stop_words, r_str):
    arr = r_str.split()

    final = []
    for word in arr:
        if word not in stop_words:
            final.append(word)

    final_str = " ".join(final)
    return final_str

ndf['rm_stop_words']=ndf['removing_'].apply(lambda x: remove_stopwords(stop_words,x))
# lambda x: ' '.join([word for word in x.split() if word not in (remove_stopwords)]


# In[1009]:


ndf.head(5)


# In[1010]:


# remove punctuation
import string
ndf['rm_punctuation'] = ndf['rm_stop_words'].apply(lambda x: re.sub(r'[^\w\s]', "", x))
#                                                  join([word for word in x.translate(string.punctuation)]))


# In[1011]:


ndf.head(5)


# In[1012]:


# remove symbols
import re
ndf['rm_symbols'] = ndf['rm_punctuation'].apply(lambda x: re.sub(r'[^\w]', " ", x))
ndf.head(5)


# In[1013]:


# remove numbers
ndf['rm_numbers'] = ndf['rm_symbols'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
ndf.head(5)


# In[1014]:


ndf['text'] = ndf['rm_numbers'].str.strip()
ndf.head(5)


# In[1015]:


ndf['final']=ndf['text'].apply(lambda x: x.lower())
ndf.head(5)


# In[1016]:


ndf.to_csv("espn_task2.csv", index=False)


# # Task 3

# In[1017]:


from sklearn.model_selection import train_test_split
import numpy as np


# In[1018]:


X_train, X_test = train_test_split(ndf["final"],
                                   random_state=1, 
                                   test_size=0.2,
                                   shuffle=True)


# In[1019]:


len(X_train)


# In[1020]:


len(X_test)


# In[1021]:


def b(X):
    b=[]
    for tweet in X:
        print(tweet)


# In[1022]:


b(X_train.head(5))


# In[1023]:


b(X_test.head(5))


# In[1024]:


def vocab(sentences):  
    words = []  
    for sentence in sentences:        
        w=sentence.split(" ")         
        words.append(w)   
    flattened_vocab= [item for sublist in words for item in sublist]
    vocab=sorted(list(set(flattened_vocab)))
    return vocab


# In[1025]:


print(vocab(X_train))
# len(tokenize(X_train))


# In[1026]:


tv=vocab(X_train)

def bow(v, allsentences):    
    bow1=[]
    for sentence in allsentences:        
        words = sentence.split(" ")        
        bag_vector = np.zeros(len(v))        
        for w in words:            
            for i,word in enumerate(v):                
                if word == w:                     
                    bag_vector[i] += 1     
        bow1.append(np.array(bag_vector).astype(int).tolist())
    return bow1
        


# In[1027]:


train_bow= bow(tv,X_train)
# print(train_bow[0])
print(len(train_bow[0]))
print(len(train_bow[1]))


# In[1028]:


test_bow= bow(tv,X_test)
# print(train_bow[0])
print(len(test_bow[0]))
print(len(test_bow[1]))


# In[1029]:


# smoothing
fea_lst_train = [[n+1 for n in sub] for sub in train_bow]
fea_lst_test = [[n+1 for n in sub] for sub in test_bow]
print(fea_lst_train[0])
print(fea_lst_test[0])


# In[1030]:


# Displaying 10 train features
df_train = pd.DataFrame(fea_lst_train)
df_train.head(10)


# In[1034]:


# Displaying 10 test features 
df_test = pd.DataFrame(fea_lst_test)
df_test.head(10)


# ## Task 3 with Scikit 

# In[1035]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(stop_words='english')
bow.fit(X_train)
len(bow.get_feature_names())


# In[1036]:


vocab = bow.get_feature_names()
vocab


# In[1037]:


# converting into vectors
features = bow.transform(X_train)
fea_array = features.toarray()


# In[1038]:


features_test = bow.transform(X_test)
fea_test_array = features_test.toarray()


# In[1039]:


print(len(fea_test_array))
print(len(fea_array))


# In[1040]:


fea_lst_train = fea_array.tolist()
fea_lst_test = fea_test_array.tolist()


# In[1041]:


#Smoothing
fea_lst_train = [[n+1 for n in sub] for sub in fea_lst_train]
fea_lst_test = [[n+1 for n in sub] for sub in fea_lst_test]


# In[1042]:


# Displaying 10 train features 
df_train = pd.DataFrame(fea_lst_train)
df_train.head(10)


# In[1043]:


# Displaying 10 test features 
df_test = pd.DataFrame(fea_lst_test)
df_test.head(10)

