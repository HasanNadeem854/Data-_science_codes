#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("D:\Data Science using pyhon\Data_Set\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# In[ ]:





# In[5]:


pd.to_numeric(df.TotalCharges,errors = 'coerce') # this command(coerce) is used to ignore all the errors 


# In[6]:


df = df[df.TotalCharges != " "] # this command is to remove all the errors from data.


# In[7]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')


# In[8]:


churn_no = df[df.Churn == 'No'] #this will give all rows for true condition
churn_no = df[df.Churn == 'No'].tenure # this will give specific column value for true condition
churn_yes = df[df.Churn == 'Yes'].tenure
plt.xlabel("tenure")
plt.ylabel("no of customers")
plt.title("histogram b/w tenure and no of customers")
plt.hist([churn_no,churn_yes],label = ("churn = yes",'churn = no'))
plt.legend()


# In[9]:


df['gender'].unique() # this is used to possibilities in sprecific column


# In[10]:


def print_unique_values(df): #now it is in form of function tha take dataframe as input.    
    for column in df:
        if df[column].dtype == 'object': # to select all the object type enteries.
            print(f'{column}:{df[column].unique()}') #f string is used here.


# In[11]:


print_unique_values(df)


# In[12]:


df.replace("No internet service","No", inplace = True)
df.replace("No phone service","No",inplace = True)
df.MultipleLines.unique()


# In[13]:


yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df[col].replace({"Yes":0,"No":1},inplace = True)


# In[22]:


df['gender'].replace({'Female':1,'Male':0},inplace=True)
df['gender']


# In[24]:


df2 = pd.get_dummies(data=df, columns=['InternetService','Contract','PaymentMethod']) # this will replace values with one and zeros
df


# In[29]:


cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale]) # this will convert all no's to 0 to 1 range
df[cols_to_scale]


# In[32]:


for new_cols in df:
    print(f'{new_cols}:{df[new_cols].unique()}')# this is used to ensure thal all the columns have unique values
    


# In[36]:


x = df.drop("Churn",axis = 'columns')
y = df.Churn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 80)


# In[44]:


import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(26,input_shape = (26,),activation = "relu"),
    keras.layers.Dense(15,activation = "relu"),
    keras.layers.Dense(1,activation = "sigmoid")
])
model.compile(optimizer = "adam",
             loss = "binary_crossentropy",
             metrics = (['accuracy']))
model.fit(x_train,y_train,epochs = 5,batch_size = 20)


# In[45]:


model.evaluate(x_test,y_test)


# In[50]:


yp = model.predict(x_test)
yp


# In[57]:


y_pre = []
for i in yp:
    if i > 0.5:
        y_pre.append(1)
    else:
        y_pre.append(0)
        
y_pre[:10]    


# In[58]:


y_test[:10]


# In[59]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[60]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




