#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# In[85]:


df_fr = pd.read_csv("df_fr_final2405.csv")
df_fr.head()


# In[86]:


df_fr["release_date"] = pd.to_datetime(df_fr["release_date"])
df_fr["release_date"] = df_fr["release_date"].apply(lambda x : x.year)


# In[104]:


var_exp = df_fr.loc[:,"averageRating"::].columns
var_y = "originalTitle"


# In[105]:


X = df_fr[var_exp]
y = df_fr[var_y]


# In[134]:


var_y = "Intouchables"


# In[135]:


df_fr_nneighbors = df_fr[df_fr["originalTitle"] != var_y]
df_fr_y = df_fr[df_fr["originalTitle"] == var_y]


# In[136]:


df_fr_nneighbors[df_fr["originalTitle"] == var_y]


# In[137]:


X_train = df_fr[var_exp]


# In[138]:


df_fr_y


# In[139]:


modelNN = NearestNeighbors(n_neighbors = 4)
modelNN.fit(X_train)


# In[127]:


X_train


# In[140]:


df_fr_y.iloc[:,2:]


# In[147]:


df_fr[290:291]


# In[148]:


df_fr[468:469]


# In[142]:


neighbors = modelNN.kneighbors(df_fr_y.iloc[:,2:])
neighbors


# In[ ]:




