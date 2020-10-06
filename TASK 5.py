#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
from sklearn.cluster import KMeans


# # Loading Iris dataset

# In[4]:


iris =pd.read_csv("Iris.csv")


# In[5]:


iris.head()


# In[6]:


iris.drop(['Id','Species'],axis=1,inplace=True)


# In[7]:


iris.head()


# In[8]:


iris.info()


# In[9]:


iris.describe()


# In[11]:


iris.isna().sum()


# In[12]:


sns.pairplot(iris)


# In[13]:


iris.corr()


# In[16]:


import matplotlib.pyplot as plt


# In[18]:


plt.figure(figsize=(8,4))
sns.heatmap(iris.corr(),annot=True,cmap='viridis')


# In[25]:


x = iris.values
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = 1, init = 'k-means++',
                    max_iter = 300,n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# ##### Preparing the dataset

# In[38]:


x=iris.iloc[:,:-1].values
y=iris.iloc[:,-1].values


# In[39]:


x.head()


# In[21]:


from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)
y


# Splitting the data into train test model

# In[22]:


from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.20,random_state=101)


# In[24]:


x_train


# In[29]:


from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=4,criterion='gini')
model.fit(x_train,x_test)


# In[33]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:




