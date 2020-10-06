#!/usr/bin/env python
# coding: utf-8

# # Task 3 Unsupervised Learning
# 

# Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision. In contrast to supervised learning that usually makes use of human-labeled data, unsupervised learning, also known as self-organization allows for modeling of probability densities over inputs. It forms one of the three main categories of machine learning, along with supervised and reinforcement learning.

# In[25]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the Data

# In[26]:


data = pd.read_csv('Iris.csv')


# Exploring the data

# In[27]:


data.head()


# In[28]:


data.info()


# In[29]:


data.isnull().sum()


# In[30]:


data.hist(figsize=(12,15))


# In[16]:


sns.pairplot(data)


# Exploring the data

# In[17]:


data.describe()


# # Finding the clusters using the K-means algorithm

# K Means Clustering tries to cluster your data into clusters based on their similarity. In this algorithm, we have to specify the number of clusters (which is a hyperparameter) we want the data to be grouped into.K-means uses an iterative refinement method to produce its final clustering based on the number of clusters defined by the user (represented by the variable K) and the dataset. For example, if you set K equal to 3 then your dataset will be grouped in 3 clusters, if you set K equal to 4 we will group the data in 4 clusters, and so on.

# In[38]:


data1 = data.iloc[:, [1,2,3,4]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data1)
    wcss.append(kmeans.inertia_)


# Plotting the elbow curve

# In[39]:


plt.plot(range(1, 11), wcss)
plt.title('Elbow Curve')
plt.xlabel('number of cluster')
plt.ylabel('within cluster sum of squares')
plt.tight_layout()
plt.xticks(range(1,11))
ab = plt.gca()
ab.spines['top'].set_visible(False)
ab.spines['right'].set_visible(False)
plt.show()


# In[37]:


data2= data.iloc[:,1:4].values
k = KMeans(n_clusters=3)
km = k.fit(data2)
predict = km.predict(data2)


# Applying KMeans cluster method

# In[24]:


plt.figure(figsize = (8,8))
plt.scatter(data1[predict == 0,0], data1[predict == 0,1],s = 100, c = 'green', label = 'iris-setosa')
plt.scatter(data1[predict == 1,0], data1[predict == 1,1],s = 100, c ='red', label = 'iris-versicolour')
plt.scatter(data1[predict == 2,0], data1[predict == 2,1],s = 100, c = 'blue', label = 'iris-virginica')


plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1],
           s = 100, c = 'yellow', label = 'Centroids')
plt.legend(loc = 'upper right')
plt.show()


# In[ ]:




