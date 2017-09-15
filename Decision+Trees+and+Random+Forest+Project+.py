
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Random Forest Project 
# 
# For this project we will be exploring publicly available data from [LendingClub.com](www.lendingclub.com). Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. We will try to create a model that will help predict this.
# 
# Lending club had a [very interesting year in 2016](https://en.wikipedia.org/wiki/Lending_Club#2016), so let's check out some of their data and keep the context in mind. This data is from before they even went public.
# 
# We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/info/download-data.action) or just use the csv already provided. It's recommended you use the csv provided as it has been cleaned of NA values.
# 
# Here are what the columns represent:
# * credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
# * purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
# * int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
# * installment: The monthly installments owed by the borrower if the loan is funded.
# * log.annual.inc: The natural log of the self-reported annual income of the borrower.
# * dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
# * fico: The FICO credit score of the borrower.
# * days.with.cr.line: The number of days the borrower has had a credit line.
# * revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
# * revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
# * inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
# * delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
# * pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

# # Import Libraries
# 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

# In[2]:

import pandas as pd
import numpy as np


# In[3]:

import matplotlib.pyplot as ply
import seaborn as sns


# In[4]:

get_ipython().magic('matplotlib inline')


# ## Get the Data
# 
# ** Use pandas to read loan_data.csv as a dataframe called loans.**

# In[5]:

loans = pd.read_csv("loan_data.csv")


# ** Check out the info(), head(), and describe() methods on loans.**

# In[7]:

loans.info()


# In[8]:

loans.head()


# In[9]:

loans.describe()


# # Exploratory Data Analysis
# 
# Let's do some data visualization! We'll use seaborn and pandas built-in plotting capabilities, but feel free to use whatever library you want. Don't worry about the colors matching, just worry about getting the main idea of the plot.
# 
# ** Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.**
# 
# *Note: This is pretty tricky, feel free to reference the solutions. You'll probably need one line of code for each histogram, I also recommend just using pandas built in .hist()*

# import random
# import numpy
# from matplotlib import pyplot
# 
# x = [random.gauss(3,1) for _ in range(400)]
# y = [random.gauss(4,2) for _ in range(400)]
# 
# bins = numpy.linspace(-10, 10, 100)
# 
# pyplot.hist(x, bins, alpha=0.5, label='x')
# pyplot.hist(y, bins, alpha=0.5, label='y')
# pyplot.legend(loc='upper right')
# pyplot.show()

# In[19]:

f1 = loans["fico"][loans["credit.policy"] ==  1]
f0 = loans["fico"][loans["credit.policy"] ==  0]


# In[26]:

f1.hist(bins = 30, edgecolor='black')
f0.hist(bins = 30, edgecolor='black')


# In[7]:




# ** Create a similar figure, except this time select by the not.fully.paid column.**

# In[29]:

NP1 = loans["fico"][loans["not.fully.paid"] ==  1]
NP0 = loans["fico"][loans["not.fully.paid"] ==  0]


# In[48]:

NP0.hist(bins = 30, edgecolor='black', color = "blue",
        label = "Credit Policy = 1", alpha = 0.6)
NP1.hist(bins = 30, edgecolor='black', color = "red",
        label = "Credit Policy = 1", alpha = 0.6)
ply.legend()


# ** Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **

# In[37]:

ply.rcParams['figure.figsize']=11,8
sns.countplot(x = "purpose", hue = "not.fully.paid", data = loans)


# ** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**

# In[42]:

sns.jointplot("fico","int.rate", data = loans, joint_kws={"s": 10}, color = "purple")


# ** Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation for lmplot() if you can't figure out how to separate it into columns.**

# In[51]:

ply.figure(figsize=(11,7))
sns.lmplot(y = 'int.rate', x = "fico", data = loans,
           hue = "credit.policy", col = "not.fully.paid")


# # Setting up the Data
# 
# Let's get ready to set up our data for our Random Forest Classification Model!
# 
# **Check loans.info() again.**

# In[43]:

loans.info()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means we need to transform them using dummy variables so sklearn will be able to understand them. Let's do this in one clean step using pd.get_dummies.
# 
# Let's show you a way of dealing with these columns that can be expanded to multiple categorical features if necessary.
# 
# **Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.**

# In[52]:

cat_feats = ['purpose']


# **Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.**

# In[57]:

final_data = pd.get_dummies(loans, columns = cat_feats, drop_first=True)


# In[58]:

final_data.info()


# notice that:
# * purpose_credit_card           9578 non-null uint8
# * purpose_debt_consolidation    9578 non-null uint8
# * purpose_educational           9578 non-null uint8
# * purpose_home_improvement      9578 non-null uint8
# * purpose_major_purchase        9578 non-null uint8
# * purpose_small_business        9578 non-null uint8
# 
# have all been created from the purpose column. Each catagory from the column was made into its own column and then labled 1/0 if the condition was met.

# ## Train Test Split
# 
# Now its time to split our data into a training set and a testing set!
# 
# ** Use sklearn to split your data into a training set and a testing set as we've done in the past.**

# In[59]:

from sklearn.cross_validation import train_test_split


# In[60]:

X = final_data.drop("not.fully.paid", axis = 1)
y = final_data["not.fully.paid"]


# In[61]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Training a Decision Tree Model
# 
# Let's start by training a single decision tree first!
# 
# ** Import DecisionTreeClassifier**

# In[62]:

from sklearn.tree import DecisionTreeClassifier


# **Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**

# In[63]:

dtree = DecisionTreeClassifier()


# In[69]:

dtree.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Create predictions from the test set and create a classification report and a confusion matrix.**

# In[74]:

prediction = dtree.predict(X_test)


# In[75]:

from sklearn.metrics import classification_report, confusion_matrix


# In[76]:

print(classification_report(y_test, prediction))


# In[77]:

print(confusion_matrix(y_test, prediction))


# ## Training the Random Forest model
# 
# Now its time to train our model!
# 
# **Create an instance of the RandomForestClassifier class and fit it to our training data from the previous step.**

# In[81]:

from sklearn.ensemble import RandomForestClassifier


# In[83]:

rfc = RandomForestClassifier(n_estimators = 300)


# In[84]:

rfc.fit(X_train, y_train)


# ## Predictions and Evaluation
# 
# Let's predict off the y_test values and evaluate our model.
# 
# ** Predict the class of not.fully.paid for the X_test data.**

# In[85]:

predict = rfc.predict(X_test)
## With the model we created we are creating an array of predictions
## based off of the test features.


# **Now create a classification report from the results. Do you get anything strange or some sort of warning?**

# In[86]:

print(classification_report(y_test, predict))


# **Show the Confusion Matrix for the predictions.**

# In[87]:

print(confusion_matrix(y_test, predict))


# **What performed better the random forest or the decision tree?**

# In this version of the exercise the random forest performed better when it came down to recall,
# but performed worse when it came down to precision.
# 
# note: recall is a measure of how the model performs when the classes are unbalanced.

# # Great Job!
