#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as st
data = pd.read_csv('gummies.csv')


# In[2]:


data


# In[3]:


# check for missing numbers
data.isna().sum()


# In[4]:


# 
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0})
data.describe()


# In[9]:


y = data['GPA']
x = data[['SAT', 'Attendance']]
x = st.add_constant(x)
model = st.OLS (y, x)
results = model.fit()
results.summary()


# In[11]:


plt.scatter(data['SAT'], data['GPA'], c=data['Attendance'], cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat_no, linewidth=2, c='#006837',label='regression_line 1')
fig = plt.plot(data['SAT'], yhat_yes, linewidth=2, c='#a50026',label='regression_line 2')
plt.xlabel('SAT', fontsize =10)
plt.ylabel('GPA', fontsize = 10)
plt.show()


# In[ ]:




