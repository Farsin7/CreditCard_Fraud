#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


card_data= pd.read_csv('creditcard.csv')
card_data.head()


# ## Data set informations

# In[4]:


card_data.info()


# In[5]:


# Checking number of missing values in each column
card_data.isnull().sum()


# In[7]:


#Distribution of legit and fraudulent transactions
card_data['Class'].value_counts()


# 0 = Normal transcation
# 1 = Fradulent transactions

# # Separating data for analysis

# In[11]:


legit = card_data[card_data.Class == 0]
fraud = card_data[card_data.Class == 1]


# In[12]:


print(legit.shape)
print(fraud.shape)


# # Statistical measures of Data

# In[13]:


legit.Amount.describe()


# In[14]:


fraud.Amount.describe()


# In[15]:


# Comparing the values of both transactions
card_data.groupby('Class').mean()


# # Undersampling to balance the data

# In[16]:


legit_sample= legit.sample(n=492)


# Concatenating two DataFrames

# In[17]:


new_dataset= pd.concat([legit_sample, fraud], axis= 0)
new_dataset.head()


# In[18]:


## Axis = 0 (Rows), 1 (Columns)


# In[19]:


new_dataset.tail()


# In[20]:


new_dataset['Class'].value_counts()


# In[21]:


new_dataset.groupby('Class').mean()


# # Splitting DataSets into Features and Targets

# In[25]:


X = new_dataset.drop('Class', axis= 1)
Y = new_dataset['Class']

X


# In[26]:


print (Y)


# # Splitting to Training and Testing Datas

# In[29]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, stratify=y, random_state=2)


# In[31]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training

# Linear Regression

# In[32]:


model = LogisticRegression()


# In[33]:


#Training model  with training data

model.fit(X_train, Y_train)


# # Model evaluation

# Accuracy Score

# In[34]:


#Training Data

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, Y_train)

print("Training Data Accuracy:", training_accuracy)


# In[35]:


#Test Data

X_test_prediction = model.predict(X_test)
test_accuracy= accuracy_score(X_test_prediction, Y_test)

print('Test Data Accuracy:', test_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




