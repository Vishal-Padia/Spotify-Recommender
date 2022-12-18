#!/usr/bin/env python
# coding: utf-8

# In[16]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()
warnings.filterwarnings('ignore')


# In[4]:


data = pd.read_csv("spotify.csv")


# ## Data Exploration

# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


df = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])
df.corr()


# ## Data Transformation

# In[8]:


from sklearn.preprocessing import MinMaxScaler


# In[10]:


datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normarization = data.select_dtypes(include=datatypes)
for col in normarization.columns:
    MinMaxScaler(col)


# In[11]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
features = kmeans.fit_predict(normarization)
data['features'] = features
MinMaxScaler(data['features'])


# ## Driver Code 

# In[14]:


class Spotify_Recommendation():
    def __init__(self,dataset):
        self.dataset = dataset
    def recommend(self,songs, amount=1):
        distance = []
        song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
        for songs in tqdm(rec.values):
            d = 0 
            for col in np.arange(len(rec.columns)):
                if not col in [1, 6, 12, 14, 18]:
                    d = d + np.absolute(float(song[col])- float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        return rec[columns][:amount]


# In[24]:


recommendations = Spotify_Recommendation(data)
recommendations.recommend("Love Story", 10)


# In[ ]:




