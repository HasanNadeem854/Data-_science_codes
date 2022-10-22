#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd # we import all the important libraries
import numpy as np    
import matplotlib.pyplot as plt
from sklearn import linear_model
import tensorflow as tf
from tensorflow import keras


# # loading of data

# In[2]:


df = pd.read_csv("D:\Data Science using pyhon\Data_Set\datafile.csv") 
df.head()


# # Cleaning of data

# In[3]:


df.drop("ID",axis = 'columns',inplace = True) # we rrmoved first column


# In[4]:


df["Gender"].replace({'F':1,'M':0},inplace = True) 


# In[5]:


df = pd.get_dummies(data = df,columns = ["State",'Product Type','Form'])
df


# In[6]:


df.drop("Channel",axis = 'columns',inplace = True)


# In[7]:


df


# # checkig of data in order to make sure that every thing is in the no form

# In[14]:


def print_unique_val(df):
    for column in df:
        if df[column].dtype == "object":
            print(f'{column}: {df[column].unique()}')
print_unique_val(df)


# # training and spliting of data

# In[15]:


from sklearn.model_selection import train_test_split
X = df.drop(["Confidence Rating",'Recommendation Score'],axis = 'columns')
y = df['Confidence Rating']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)
X_train


# # fitting the model( for classification) on data

# In[12]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB
model = GaussianNB()
model.fit(X_train,y_train)
predicted_y= model.predict(X_test)


# In[16]:


from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test,predicted_y))


# In[19]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=predicted_y)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




