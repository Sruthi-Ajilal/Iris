#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.linear_model import LinearRegression


# In[4]:


iris=datasets.load_iris()
print(iris)


# In[5]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[6]:


df


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


plt.style.use('_mpl-gallery-nogrid')

# make data: correlated + noise
np.random.seed(1)
x = np.random.randn(5000)
y = 1.2 * x + np.random.randn(5000) / 3

# plot:
fig, ax = plt.subplots()

ax.hist2d(x, y, bins=(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1)))

ax.set(xlim=(-2, 2), ylim=(-3, 3))

plt.show()


# In[10]:


plt.style.use('_mpl-gallery')

# make data
np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 200)

# plot:
fig, ax = plt.subplots()

ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 56), yticks=np.linspace(0, 56, 9))

plt.show()


# In[11]:


plt.xlabel("sepal length")
plt.ylabel("Sepal width")
plt.title("Iris")
plt.plot(df)
np.random.seed(0)
x=4+np.random.normal(0,1.5,200)


# In[12]:


sns.pairplot(df)


# In[13]:


sns.displot(df)


# In[14]:


sns.jointplot(df)


# In[15]:


sns.boxplot(df)


# In[16]:


#df=["sepal length(cm)","sepal width(cm)"]
x="sepal length(cm)","petal width (cm)"
y="sepal width(cm)","petal width (cm)"
plt.xlabel("sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Iris")
plt.plot(df)
plt.scatter(x,y)


# In[18]:


sns.lmplot(data=df, x="sepal length (cm)", y="sepal width (cm)")


# In[19]:


x=df["sepal length (cm)"].values
y=df["sepal width (cm)"].values


# In[27]:


from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.50,random_state=0)


# In[28]:


x_train


# In[29]:


y_train


# In[30]:


x_test


# In[31]:


y_test


# In[32]:


x_train=x_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_train=y_train.reshape(-1,1)
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[33]:


y_predict=reg.predict(x_test)
y_predict


# In[35]:


reg.score(x_train,y_train)*100



# In[36]:


reg.score(x_test,y_predict)*100


# In[ ]:




