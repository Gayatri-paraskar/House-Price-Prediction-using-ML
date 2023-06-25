#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries and Dataset

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
dataset = pd.read_csv("House.csv")
print(dataset)


# ### dimension of the dataset

# In[3]:


dataset.shape   


# ### Data Preprocessing
# 

# In[6]:


obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# ### Exploratory Data Analysis

# In[7]:


plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(),
			cmap = 'BrBG',
			fmt = '.2f',
			linewidths = 2,
			annot = True)


# ### To analyze the different categorical features we have drawn a barplot.

# In[8]:


unique_values = []
for col in object_cols:
 unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)


# ### To findout the actual count of each category we can plot the bargraph of each four features separately.

# In[9]:


plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
	y = dataset[col].value_counts()
	plt.subplot(11, 4, index)
	plt.xticks(rotation=90)
	sns.barplot(x=list(y.index), y=y)
	index += 1


# ### Data Cleaning

# ####  Drop  Id Column As Id Column will not be participating in any prediction.

# In[10]:


dataset.drop(['Id'],
			axis=1,
			inplace=True)


# #### Replacing SalePrice empty values with their mean values to make the data distribution symmetric.

# In[11]:


dataset['SalePrice'] = dataset['SalePrice'].fillna(
dataset['SalePrice'].mean())


# #### Drop records with null values using dropna()

# In[12]:


new_dataset = dataset.dropna()


# #### Checking features which have null values in the new dataframe 

# In[13]:


new_dataset.isnull().sum()


# ### OneHotEncoder – For Label categorical features
# - One hot Encoding is the best way to convert categorical data into binary vectors.
# - This maps the values to integer values. 
# - By using OneHotEncoder, we can easily convert object data into int.
# - firstly we have to collect all the features which have the object datatype. 
# - To do so, we will make a loop.

# In[14]:


from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ',
	len(object_cols))


# ####  once we have a list of all the features. We can apply OneHotEncoding to the whole list.

# In[15]:


OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)


# ### Splitting Dataset into Training and Testing
# #### X and Y splitting (i.e. Y is the SalePrice column and the rest of the other columns are X)
# 
# 

# In[31]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
X, Y, train_size=0.8, test_size=0.2, random_state=0)


# ### Model and Accuracy
# #### As we have to train the model to determine the continuous values, so we will be using these regression models.
# 
# - SVM-Support Vector Machine
# - Random Forest Regressor
# - Linear Regressor
# #### And To calculate loss we will be using the mean_absolute_percentage_error module. It can easily be imported by using sklearn library.

# ### SVM – Support vector Machine
# - SVM can be used for both regression and classification model. It finds the hyperplane in the n-dimensional plane. 

# In[32]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)

print(mean_absolute_percentage_error(Y_valid, Y_pred))


# ### Random Forest Regression
# - Random Forest is an ensemble technique that uses multiple of decision trees and can be used for both regression and classification tasks.

# In[33]:


from sklearn.ensemble import RandomForestRegressor

model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)

mean_absolute_percentage_error(Y_valid, Y_pred)


# ### Linear Regression
# - Linear Regression predicts the final output-dependent value based on the given independent features.
# - Like, here we have to predict SalePrice depending on features like MSSubClass, YearBuilt, BldgType, Exterior1st etc.

# In[34]:


from sklearn.linear_model import LinearRegression
 
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))


# ### CatBoost Classifier
# - CatBoost is a machine learning algorithm implemented by Yandex and is open-source.
# - It is simple to interface with deep learning frameworks such as Apple’s Core ML and Google’s TensorFlow.
# - Performance, ease-of-use, and robustness are the main advantages of the CatBoost library. 

# In[47]:


from catboost import CatBoostRegressor
cb_model = CatBoostRegressor()
cb_model.fit(X_train, Y_train)
preds = cb_model.predict(X_valid)
from sklearn.metrics import r2_score
cb_r2_score=r2_score(Y_valid, preds)
print(cb_r2_score)


# ### Project is made by: Gayatri Paraskar.
