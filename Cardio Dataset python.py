#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[294]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder, StandardScaler , PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier ,VotingClassifier , AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score , accuracy_score 
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading and Exploring Data

# ## Loading Data

# In[295]:


Cardio_dataset=(
    pd.read_csv(r"C:\Users\Administrator\Desktop\my_lab\CVD dataset\cardio_train.csv" , sep=";")
)
Cardio_dataset=Cardio_dataset[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']]
Cardio_dataset.shape


# ## EDA

# In[296]:


Cardio_dataset.head()


# In[297]:


Cardio_dataset.info()
# no null value


# In[298]:


Cardio_dataset.describe().T


# In[299]:


Cardio_dataset.drop_duplicates(keep="first",inplace=True) 


# In[300]:


Cardio_dataset.shape


# ### Data Visualization

# #### plotting numerical columns 

# In[301]:


def plot_numerical_col (dataset , var):
    return sns.displot(data=dataset , x=var , kde=True)
numeric_columns=['age' , 'height' , 'weight']
for col in numeric_columns:
    plot_numerical_col(Cardio_dataset , col)


# Before I feed the dataset into the model, it is essential to clean the diastolic and systolic datasets because the min and max values based on the described table is an outliers and should be removed and checked the value. 
# First, I show them by boxplot and then remove the outliers.

# In[302]:


sns.scatterplot(data = Cardio_dataset[(Cardio_dataset["ap_hi"] < 400) & (Cardio_dataset["ap_lo"] < 400) & (Cardio_dataset["ap_hi"] > 10) & (Cardio_dataset["ap_lo"] > 10)], x = "ap_hi", y = "ap_lo")


# In[303]:


Cardio_dataset["pulsus_reversus"]= Cardio_dataset['ap_hi'] < Cardio_dataset['ap_lo']
Cardio_dataset.pulsus_reversus.value_counts()


# In[304]:


# Outlier Detection and Relationship
sns.scatterplot(y=Cardio_dataset['weight'],x=Cardio_dataset['height'],hue=Cardio_dataset['cardio'])
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Relationship between Height, Weight and Cardio')
plt.show()


# #### Correlation matrix

# In[305]:


correlation_matrix = Cardio_dataset.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Variable Correlation Heatmap')
plt.show()


# #### plotting categorical columns

# Using the map method to change numerical values to categorical ones can make your bar chart clearer and more interpretable. 

# In[306]:


Cardio_dataset["gender"]=Cardio_dataset["gender"].map({1:'Man' , 2:'Woman'})
Cardio_dataset["cholesterol"]=Cardio_dataset["cholesterol"].map({1:"normal" , 2:"above normal" , 3:"well above normal"})
Cardio_dataset["gluc"]=Cardio_dataset["gluc"].map({1:"normal" , 2:"above normal" , 3:"well above normal"})
Cardio_dataset["smoke"]=Cardio_dataset["smoke"].map({0:"no" , 1:"yes"})
Cardio_dataset["alco"]=Cardio_dataset["alco"].map({0:"no" , 1:"yes"})
Cardio_dataset["active"]=Cardio_dataset["active"].map({0:"no" , 1:"yes"})


# In[307]:


categorical_var_=Cardio_dataset.loc[: , ['gender','cholesterol', 'gluc', 'smoke', 'alco', 'active' ,'pulsus_reversus']]
plt.figure(figsize=(10,8))
sns.countplot(x="variable", hue="value",data= pd.melt(categorical_var_))


# In[308]:


sns.catplot(data=Cardio_dataset.melt(id_vars='cardio', value_vars=['gender','cholesterol', 'gluc', 'smoke', 'alco', 'active' ,'pulsus_reversus']), x="variable", hue="value", col="cardio", kind="count")


# # Data Preprocessing

# ## Removing outlier

# In[309]:


def remove_outlier(dataset , var):
       return dataset[(dataset[var] >= 10) & (dataset[var] <= 400)] 
    
Cardio_dataset=remove_outlier(Cardio_dataset, 'ap_lo')
Cardio_dataset=remove_outlier(Cardio_dataset, 'ap_hi')


def remove_outlier_height (dataset , height):
     return dataset[(dataset[height] >= 75) & (dataset[var] <= 250)]

Cardio_dataset=remove_outlier(Cardio_dataset, 'height')

    
def remove_outlier_weight(dataset , weight):
    return dataset[(dataset[var] >= 25) & (dataset[var] <= 300)]
Cardio_dataset=remove_outlier(Cardio_dataset, 'weight')


# In[310]:


Cardio_dataset.shape


# In[311]:


Cardio_dataset.pulsus_reversus.value_counts()


# ## Adding New Features

# In[312]:


def calculate_bmi(dataset , weight, height):
    height_m = dataset[height]/100
    bmi = dataset[weight] / (height_m ** 2)
    return bmi

Cardio_dataset["bmi"]=calculate_bmi(Cardio_dataset , "weight", "height")


def calculate_blood_pressure(dataset, systolic , diastolic):
    return dataset[systolic]/dataset[diastolic]

Cardio_dataset["BP"]=calculate_blood_pressure(Cardio_dataset ,"ap_hi" , "ap_lo" )

def Pulsus_reversus(dataset, systolic , diastolic):
    
    return Cardio_dataset[systolic] < Cardio_dataset[diastolic]

Cardio_dataset['Pulsus_reversus']=Pulsus_reversus(Cardio_dataset ,'ap_hi' , 'ap_lo' )

# change the age feature from days to the year

def days_to_year(dataset , col):
    return round(dataset[col]/365 , 2)
Cardio_dataset["Age"]=days_to_year(Cardio_dataset , "age")


# ## Split_test_train

# In[313]:


Cardio_dataset["cardio"].value_counts(normalize=True) * 100


# In[324]:


y=Cardio_dataset["cardio"]
X=Cardio_dataset.drop('cardio' , axis=1)
X_train ,  X_test ,y_train, y_test=train_test_split(X , y, test_size=0.2 , random_state=42)
X_train.shape


# ## ColumnTransformer

# In[315]:


number_list=Cardio_dataset[['age', 'height', 'weight', 'ap_hi', 'ap_lo','bmi', 'BP', 'Age']]
list_number=list(number_list)
list_cat_ordinal=list(["cholesterol", "gluc"])
list_cat_binary=list(["gender" , "smoke" , "alco" , "active"])

coltransform=ColumnTransformer([
    ('std' , StandardScaler() , list_number), 
    ('ordinal' ,OrdinalEncoder() , list_cat_ordinal ), 
    ('1hot' , OneHotEncoder() , list_cat_binary)
    
])
X_train_pre=coltransform.fit_transform(X_train)


# # Model_Building

# In[320]:


lr=LogisticRegression(max_iter=100)
rf=RandomForestClassifier(n_estimators=51,max_depth=10)
xgb= XGBClassifier(n_estimators=150,gamma= 0.24, max_depth=4, learning_rate=0.13,reg_lambda=50.0, scale_pos_weight=1)
knn=KNeighborsClassifier(weights = 'uniform', n_neighbors = 300,leaf_size = 1)

models=[lr , rf, xgb ,knn]
for  model in models:
    model.fit(X_train_pre , y_train)
    y_pred=cross_val_predict(model , X_train_pre , y_train , cv=5)
    precision=precision_score(y_train,y_pred)
    recall=recall_score(y_train,y_pred)
    print(f'Model :{model.__class__.__name__}')
    print(f'Precisio score : {precision}')
    print(f'Recall score : {recall}')
    print('-----------------------------')


# In[321]:


y_probs = cross_val_predict(xgb, X_train_pre , y_train, cv=5, method='predict_proba')[:, 1]

threshold = 0.3 

y_pred_ = (y_probs >= threshold).astype(int)

recall = recall_score(y_train, y_pred_)
precision = precision_score(y_train, y_pred_)
confusionmatrix=confusion_matrix(y_train, y_pred_)
accuracy = accuracy_score(y_train  , y_pred_)                                  
print(f"confusion matrix:")
print(confusionmatrix)
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"accuracy: {accuracy}")


# In[325]:


X_test_pre = coltransform.transform(X_test)
X_test.shape


# In[334]:


y_pred_1 = xgb.predict(X_test_pre)


# In[335]:


recall_score(y_test, y_pred_1)

