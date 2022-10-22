#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# In[3]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[4]:


X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[5]:


model_lr  = LogisticRegression(solver='liblinear',multi_class='ovr')
model_lr.fit(X_test,y_test)
model_lr.score(X_test, y_test)


# In[6]:


model_svc = SVC()
model_svc.fit(X_test,y_test)
model_svc.score(X_test, y_test)


# In[7]:


rfc = RandomForestClassifier(n_estimators = 10)
rfc.fit(X_test,y_test)
rfc.score(X_test, y_test)


# In[8]:


def get_score(model,X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


# In[9]:


folds = StratifiedKFold(n_splits = 3)


# In[10]:


scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index],                                        digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(model_svc,X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(rfc,X_train, X_test, y_train, y_test))


# In[11]:


scores_logistic


# In[12]:


scores_svm


# In[13]:


scores_rf


# In[18]:


scores1 = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'),digits.data,digits.target,cv=10)
np.average(scores1)


# In[22]:


scores2 = cross_val_score(rfc,digits.data,digits.target,cv=10)
np.average(scores2)


# In[21]:


scores3 = cross_val_score(model_svc,digits.data,digits.target,cv=10)
np.average(scores3)


# In[ ]:




