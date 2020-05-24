#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


#### Scikit learn se model ko train krna
#!pip install sklearn
#!pip install pickle
import pickle


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


model = LinearRegression()


# In[11]:


## To calculate R square(r2)
from sklearn.metrics import r2_score


# In[12]:


df = pd.read_csv('loanf.csv')


# In[13]:


df.keys()


# In[14]:


df.shape


# In[15]:


x = df[["FICO.Score","Loan.Amount"]].values
y = df["Interest.Rate"].values


# In[16]:


x


# In[17]:


y


# In[18]:


model = model.fit(x,y)


# In[19]:


model


# In[20]:


c = model.intercept_
c


# In[21]:


m = model.coef_
m


# In[22]:


y1 = m*[20,4000]+ c
y1


# In[23]:


model.predict([[20,4000]])


# In[24]:


model.predict([[1000,120000]])


# In[25]:


# function to calculate interest rate


# In[27]:


# to remove decimals from interst rate convert it to int
def InterestRate(FICO,Amount):
    rate = model.predict([[FICO,Amount]])
    print(f"The interest for loan amount {Amount} with a FICO score of {FICO} is {int(rate[0])}%")
    return rate


# In[28]:


ans = InterestRate(600,12000)


# In[29]:


int(ans[0])


# In[30]:


pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:




