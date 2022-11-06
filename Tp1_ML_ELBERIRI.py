#!/usr/bin/env python
# coding: utf-8

# In[39]:


#Pour lire les fichiers
import pandas as pd
import numpy as np

#Pour créer notre diagramme
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')

import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[56]:


def read_pickle_file(file):
    pickle_data = pd.read_pickle(file)
    return pickle_data



neg=read_pickle_file(r"C:\Users\naouf\OneDrive\Bureau\ML\imdb_raw_neg.pickle")
pos=read_pickle_file(r"C:\Users\naouf\OneDrive\Bureau\ML\imdb_raw_pos.pickle")



df_neg=pd.DataFrame(neg)
df_neg.columns=["review"]
df_neg["sentiment"]="Neg"

df_pos=pd.DataFrame(pos)
df_pos.columns=["review"]
df_pos["sentiment"]="Pos"


df=pd.concat([df_neg,df_pos])
df=df.sample(frac=1)


# In[57]:


sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")

#On constate que les avis sont bien répartis de façons équivalente


# In[58]:


for i in range(5):
    print("Review: ", [i])
    print(df['review'].iloc[i], "\n")
    print("Sentiment: ", df['sentiment'].iloc[i], "\n\n")
  


# In[60]:


def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count

df['word count'] = df['review'].apply(no_of_words)
df.head()

#Cette fonction nous permettra de calculer le nombre de mots dans chaque commentaires, on l'utilisera pour comparer
# le nombre de mots dans chaque commentaires après avoir retirer les éventuels doublon qui risquent de fausser notre modèle


# In[75]:


df.sentiment.replace("Pos", 1, inplace=True)
df.sentiment.replace("Neg", 2, inplace=True)

df.head()


# In[76]:


def data_processing(text):
    text= text.lower()
    text = re.sub('', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

df.review = df['review'].apply(data_processing)


# In[100]:


duplicated_count = df.duplicated().sum()


# In[78]:


df = df.drop_duplicates('review')


# In[79]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

df.review = df['review'].apply(lambda x: stemming(x))
df['word count'] = df['review'].apply(no_of_words)
df.head()

#En comparant le nombre de mots initials avec celui que nous avons à présent on voit bien que les doublons on bien étais retirés


# In[84]:


pos_reviews =  df[df.sentiment == 1]


neg_reviews =  df[df.sentiment == 2]


# In[89]:


#Maintenant que les doublon sent retirés, regardons quels sont les 10 mots qui impacteront le plus notre modèle


# In[88]:


#Sur le commentaire positifs

from collections import Counter
count = Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(10)

pos_words = pd.DataFrame(count.most_common(10))
pos_words.columns = ['word', 'count']
pos_words.head()


# In[90]:


#Sur les commentaires négatifs

count = Counter()
for text in neg_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)


neg_words = pd.DataFrame(count.most_common(15))
neg_words.columns = ['word', 'count']
neg_words.head()


# In[91]:


#On constate que les mots sont les même, mais de taille différentes.


# In[93]:


#Construisons maintenant notre modèle

X = df['review']
Y = df['sentiment']

vect = TfidfVectorizer()
X = vect.fit_transform(df['review'])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[96]:


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[97]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

#Notre modèle atteint une précision de 89% sur 30% des données de test


# In[98]:


#Réalisons notre matrices de confusions

print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


# In[99]:


#Pour rendre notre modèle plus robuste on pourrait utiliser d'autre méthode d'évaluation comme
#la régression multinomiale ou une SVM

