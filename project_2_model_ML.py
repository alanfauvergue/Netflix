#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# In[67]:


df_fr = pd.read_csv("df_fr_final2405.csv")
df_fr.head()


# In[69]:


df_fr.iloc[:,10:] = df_fr.iloc[:,10:].apply(lambda x : x*2)


# In[72]:


df_fr["release_date"] = pd.to_datetime(df_fr["release_date"])
df_fr["release_date"] = df_fr["release_date"].apply(lambda x : x.year)


# In[73]:


var_exp = df_fr.loc[:,"averageRating"::].columns
var_y = "originalTitle"


# In[74]:


X = df_fr[var_exp]
y = df_fr[var_y]


# In[75]:


var_y = "Intouchables"


# In[76]:


df_fr_nneighbors = df_fr[df_fr["originalTitle"] != var_y]
df_fr_y = df_fr[df_fr["originalTitle"] == var_y]


# In[77]:


df_fr_nneighbors[df_fr["originalTitle"] == var_y]


# In[78]:


X_train = df_fr_nneighbors[var_exp]


# In[79]:


df_fr_y


# In[80]:


modelNN = NearestNeighbors(n_neighbors = 4)
modelNN.fit(X_train)


# In[81]:


X_train


# In[82]:


df_fr_y.iloc[:,2:]


# In[83]:


df_fr[290:291]


# In[84]:


df_fr[468:469]


# In[85]:


neighbors = modelNN.kneighbors(df_fr_y.iloc[:,2:])
proposition_film = list(neighbors[1][0][0:])


# In[86]:


proposition_film


# In[54]:


for film in proposition_film:
    print(df_fr.loc[film,"originalTitle"])

