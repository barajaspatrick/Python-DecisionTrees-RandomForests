
# coding: utf-8

# # Decision Trees and Random Forests with Python

# In[31]:

import pandas as pd
import numpy as np


# In[32]:

import matplotlib.pyplot as ply
import seaborn as sns


# In[33]:

get_ipython().magic('matplotlib inline')


# In[34]:

df = pd.read_csv("kyphosis.csv")


# In[35]:

df.head(4)


# In[36]:

df.info()


# In[37]:

sns.pairplot(df, hue = "Kyphosis")


# In[38]:

from sklearn.cross_validation import train_test_split


# In[39]:

X = df.drop("Kyphosis", axis = 1)


# In[40]:

y = df["Kyphosis"]


# In[41]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[42]:

from sklearn.tree import DecisionTreeClassifier


# In[43]:

dtree = DecisionTreeClassifier()


# In[44]:

dtree.fit(X_train, y_train)


# In[45]:

predictions = dtree.predict(X_test)


# In[46]:

from sklearn.metrics import classification_report, confusion_matrix


# In[47]:

print(confusion_matrix(y_test, predictions))


# In[48]:

print(classification_report(y_test, predictions))


# ## Now we want to Compare these results with that of a random forest classifier

# In[55]:

from sklearn.ensemble import RandomForestClassifier


# In[56]:

rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)


# In[57]:

print(confusion_matrix(y_test, rfc_pred))
print("\n")
print(classification_report(y_test, rfc_pred))


# innitialy difficult to tell what did better, but that depends on what we value: precision vs recall
