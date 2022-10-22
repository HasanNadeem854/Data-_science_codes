#!/usr/bin/env python
# coding: utf-8

# In[55]:


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


# In[56]:


from sklearn.datasets import load_iris


# In[57]:


iris = load_iris()
df = pd.DataFrame(iris.data,columns = iris.feature_names)
df["flower"] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
df.head()


# In[58]:


X = iris.data
y=iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

model = SVC(C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale')
model.fit(X_train,y_train)
model.predict(X_test)


# In[59]:


cross_val_score(SVC( C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale'),iris.data,iris.target,cv=5)


# In[19]:


cross_val_score(SVC( C=1.0,
    kernel='linear',
    degree=3,
    gamma='scale'),iris.data,iris.target,cv=5)


# In[20]:


cross_val_score(SVC( C=10,
    kernel='rbf',
    degree=3,
    gamma='auto'),iris.data,iris.target,cv=5)


# In[60]:


kernel = ['rbf','linear']
c = [1,10,20]
avg_score = {}
for kval in kernel:
    for cval in c:
        cv_scores = cross_val_score(SVC( C=cval,kernel=kval),iris.data,iris.target,cv=5)
        avg_score[kval +'-'+ str(cval)] =np.average(cv_scores)
avg_score        


# from sklearn.model_selection import GridSearchCV
# model = GridSearchCV(SVC,{'C':[1.0,5],
#     'kernel':['rbf','linear']},cv=5,return_train_score = False)
# 

# In[61]:


from sklearn.model_selection import GridSearchCV 
clf = GridSearchCV(SVC(gamma = 'auto'),{'C':[1.0,5], 'kernel':['rbf','linear']},cv=5,return_train_score = False)
clf.fit(iris.data, iris.target)
clf.cv_results_


# In[48]:


cv_result = pd.DataFrame(clf.cv_results_)
cv_result.head()


# In[62]:


df[['param_C','param_kernel','mean_test_score']]


# In[63]:


model_params={
    'svm':{
        'model':SVC(gamma = "auto"),
        'params':{
            "c":[1,5],
            "kernel":["linear","rbf"]
        }  
    },'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
             }


# In[64]:


scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[ ]:





# In[ ]:




