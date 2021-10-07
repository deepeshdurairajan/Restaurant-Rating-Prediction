#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression


# In[2]:


cd Downloads


# In[6]:


df=pd.read_csv("zomato.csv")


# In[7]:


df.head()


# In[8]:


df.shape 


# In[9]:


col = ['url', 'address', 'phone', 'dish_liked', 'menu_item']
df.drop(col, inplace=True, axis=1)


# In[10]:


df.info()


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.isnull().sum()


# In[14]:


def filter_rate(val):
    if (val=='-' or val=='NEW'):
        return np.nan
    else:
        val=str(val).split('/')[0] 
    return float(val)

df['rate']=df['rate'].apply(filter_rate)

df['rate']


# In[15]:


df['rate'].fillna(df['rate'].mean(),inplace=True)


# In[17]:


df.dropna(inplace=True)


# In[18]:


df.isnull().sum()


# In[19]:


df['Number_of_cuisines_offered'] = df['cuisines'].apply(lambda x : len(x.split(',')))


# In[20]:


df


# In[109]:


plt.figure(figsize=(20,10))
sns.countplot(x='location', data=df)
plt.title('Count of Restaurants at each Location', fontsize=18, fontweight='bold')
plt.xticks(rotation=90)
plt.show()


# In[112]:


a = df.groupby('Number_of_cuisines_offered').agg({'rate':'mean'})
plt.rcParams["figure.figsize"] = (12,8)
a.plot(kind='bar', color='#FFA500')
plt.title('Average ratings based on number of cuisines offered', fontsize=20, fontweight='bold')
plt.ylabel('average ratings')
plt.legend()
plt.show()


# In[111]:


plt.figure(figsize=(15,7))
chains = df['name'].value_counts()[:20]
sns.barplot(x=chains, y=chains.index, palette='Set2')
plt.xlabel("Number of outlets", size=15)
plt.ylabel("Name of Restaurants", size=15)
plt.title("Most famous restaurant chains", fontsize=20, fontweight='bold')
plt.show()


# In[25]:


plt.figure(figsize=(10,8))
df_cuisines = df['cuisines'].value_counts()[:15]
sns.barplot(x = df_cuisines.values, y=df_cuisines.index)
plt.title('Most popular cuisines in Bangalore', fontsize=20, fontweight='bold')
plt.show()


# In[115]:


sns.countplot(df['online_order'])
fig=plt.gcf()
fig.set_size_inches(8,8)
plt.title('Restaurants delivering online or not')


# In[36]:


df.name = df.name.apply(lambda x:x.title())
df.online_order.replace(('Yes','No'),(True, False),inplace=True)
df.book_table.replace(('Yes','No'),(True, False),inplace=True)
df.head()


# In[46]:


df=df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})

df.columns


# In[94]:


def Encode(df):
    for column in df.columns[~df.columns.isin(['rate', 'cost', 'votes'])]:
        df[column] = df[column].factorize()[0]
    return df

data = Encode(df.copy())
data.head()


# In[102]:


x = data.iloc[:,[1,2,4,5,6,7,8,12]]
y = data['rate']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=350)
x_train.head()


# In[103]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test) 

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[104]:


from sklearn.tree import DecisionTreeRegressor


DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[105]:


from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)

r2_score(y_test,y_predict)


# In[106]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)

y_predict=gbr.predict(x_test)



r2_score(y_test,y_predict)


# In[107]:


from sklearn.ensemble import  ExtraTreesRegressor
ETree=ExtraTreesRegressor(n_estimators = 100)
ETree.fit(x_train,y_train)
y_predict=ETree.predict(x_test)


r2_score(y_test,y_predict)


# In[108]:


models = pd.DataFrame({
    'Model' : ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boost','Extra Tree Regressor'],
    'Score' : [reg.score(x_test, y_test), DTree.score(x_test, y_test), RForest.score(x_test, y_test),
               gbr.score(x_test, y_test),ETree.score(x_test, y_test)]
})


models.sort_values(by = 'Score', ascending = False)


# In[ ]:




