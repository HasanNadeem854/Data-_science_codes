#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data,columns = iris.feature_names)
df["target"] = iris.target
df.head()


# In[4]:


df["flower names"] = df['target'].apply(lambda x: iris.target_names[x])
df.head()


# In[4]:


df[df.target==0].head() 


# In[5]:


df[df.target == 1].head()


# In[6]:


df[df.target == 2].head()


# In[7]:


df0 = df[:50]
df1 = df[50:100]


# In[8]:


df2 = df[100:]


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("sepal length(cm)")
plt.ylabel("sepal width(cm)")
plt.scatter(df1["sepal length (cm)"],df1["sepal width (cm)"],color = "red", marker = "*")
plt.scatter(df0["sepal length (cm)"],df0["sepal width (cm)"],color = "green", marker = "+")


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("petal length(cm)")
plt.ylabel("petal width(cm)")
plt.scatter(df1["petal length (cm)"],df1["petal width (cm)"],color = "red", marker = "*")
plt.scatter(df0["petal length (cm)"],df0["petal width (cm)"],color = "green", marker = "+")


# In[29]:


from sklearn.model_selection import train_test_split
X = df.drop(['target','flower names'], axis = 'columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 


# In[31]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model.predict(X_test)


# In[32]:


y_test


# In[35]:


model.score(X_test,y_test)


# In[ ]:




