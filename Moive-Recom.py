#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')


# In[3]:


credits = pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies


# In[5]:


movies.info()


# In[6]:


movies.isnull().sum()


# In[7]:


credits


# In[8]:


movies = movies.merge(credits,on='title')


# In[9]:


movies


# # Selected comluns
# genres
# id 
# keywords
# overview
# title
# movie_id
# cast

# In[10]:


movies = movies[['genres','keywords','overview','title','movie_id','cast','crew']]


# In[11]:


movies


# # preprocessing

# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


#[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'


# In[17]:


import ast


# In[18]:


def convert(obj):
    L =[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[19]:


movies['genres']=movies['genres'].apply(convert)


# In[20]:


movies.head()


# In[21]:


movies['keywords'].apply(convert)


# In[22]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[23]:


movies.head()


# In[24]:


movies.cast[0]


# In[25]:


def convert3(obj):
    L =[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!= 3:
            L.append(i['name'])
            counter+=1

    return L


# In[26]:


movies['cast'].apply(convert3)


# In[27]:


movies['cast'] = movies['cast'].apply(convert3)


# In[28]:


movies.head()


# In[29]:


movies['crew'][0]


# In[30]:


def fetch_director(obj):
    L =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[31]:


movies['crew'].apply(fetch_director)


# In[32]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[33]:


movies


# In[34]:


movies['overview'][0]


# In[35]:


movies['overview'].apply(lambda x:x.split())


# In[36]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[37]:


movies.head()


# In[38]:


movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])


# In[39]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['overview'] = movies['overview'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])


# In[40]:


movies.head()


# In[41]:


movies ['tags'] = movies['overview'] + movies['keywords'] + movies['crew'] + movies['cast'] + movies['genres']


# In[42]:


new_df = movies[['movie_id','title','tags']]


# In[43]:


new_df


# In[44]:


new_df['tags'].apply(lambda x:" ".join(x))


# In[45]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[46]:


new_df.head()


# In[47]:


new_df['tags'][100]


# In[48]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[49]:


new_df.head()


# In[50]:


new_df['tags'][0]


# In[51]:


new_df['tags'][2]


# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words = 'english')


# In[53]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[54]:


vectors


# In[55]:


(cv.get_feature_names())


# In[ ]:





# In[ ]:





# In[56]:


get_ipython().system('pip install nltk')


# In[57]:


['Loved','Loving','love']
['Love','Love','love']


# In[58]:


ps.stem('Love')


# In[ ]:


import nltk


# In[ ]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[ ]:


def stem():
    y = []
    for i in text.split():
        
        y.append(ps.stem(i))
    string = " ".join(y)


# In[ ]:





# In[ ]:


new_df['tags'][0]


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


similarity=cosine_similarity(vectors)


# In[ ]:


similarity[1]


# In[ ]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])[1:6]
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


# In[ ]:


recommend('Superman')


# In[62]:


import pickle


# In[ ]:


pickle.dump(new_df,open('movie_list.pkl','wb'))
#pickle.dump(similarity,open('similarity.pkl','wb'))


# In[63]:


pickle.dump(new_df.to_dict(),open('movie_dict_list.pkl','wb'))

