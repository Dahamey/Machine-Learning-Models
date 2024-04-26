#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;">Saving a Machine Learning Model</h1>

# <u>**Author</u> :** [Younes Dahami](https://www.linkedin.com/in/dahami/)

# # Introduction
# 
# If we need to solve a familiar problem on a different set of data with the same characteristics, we can reuse the machine learning model we made before. However, to use it again, we have to save it first. In this guide, I'll show you how to save a machine learning model using Python.

# # Why Should We Save a Machine Learning Model ?
# 
# After we've trained a machine learning model, saving it allows us to reuse it later for testing on new data or for comparing its performance against other models. We can also deploy a saved model in our final application.
# 
# There are several ways to save a machine learning model. If you're new to this, the following section will show you a simple method for saving and reusing a machine learning model using Python.

# # Steps for Saving a Machine Learning Model
# 
# Here are the steps for saving a machine learning model :
# 
#    * 1) Train the model
#    * 2) Convert it into a byte stream (pickling)
#    * 3) Save the byte stream as a binary file
#    
# In Python, converting a machine learning model into a byte stream is called **pickling**. If you want to use the saved model again, you'll need to convert the byte stream file back into a usable machine learning model, which is called **unpickling**.
# 
# Now, let's look at how to save a machine learning model. First, I'll train a linear regression model, and then I'll save it using the `pickle` method in Python.

# In[24]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn.metrics

df_raw = pd.read_csv("student-mat.csv", usecols= ["G1", "G2", "G3", "studytime", "failures", "absences"], delimiter=";")
df = df_raw.copy()
df.head()


# In[8]:


# Predicting G3 using all the other features :
X = df.drop(["G3"], axis= 1)
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=25)


# In[9]:


# Building the model
lr = LinearRegression()
model = lr.fit(X_train, y_train)


# Let's try now to save the model :

# In[29]:


import pickle
with open("pickle_model", "wb") as file :
    pickle.dump(model, file)


# After saving the model, you'll notice a new file named 'pickle_model' in the same directory as your Python file. If you wish to use the saved model for predicting values on the test set, you can execute the code provided below :

# In[47]:


with open("pickle_model", "rb") as file :
    model = pickle.load(file)
    
y_pred = model.predict(X_test)

for i in range(len(y_pred)) :
    print(f"from the feature values : {X_test.values[i]} :\n\
    We predicted {y_pred[i]} whereas the ground truth is equal to {y_test.values[i]}\n")


# # Summary
# 
# That's the process for saving a machine learning model as a byte stream, making it usable for future tasks on new datasets. I hope you found this guide helpful on how to save a machine learning model using Python.

# # Change Log
# 
# | Date (DD-MM-YYYY) | Version | Changed By      | Change Description      |
# | ----------------- | ------- | -------------   | ----------------------- |
# | 15-03-2024       | 1.0     | Younes Dahami   |  initial version |
# 

# In[ ]:




