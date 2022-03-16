#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive # Import a library named google.colab
drive.mount('/content/drive', force_remount=True) # mount the content to the directory `/content/drive`


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Netflix_Appetency_Prediction')


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train_df=pd.read_csv('train.csv')


# In[ ]:


test_df=pd.read_csv('test.csv')


# In[ ]:


id_list=test_df['id']


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# #1.Find categorical variables and numerical variables

# In[ ]:


categorical_var=[column for column in train_df.columns if (train_df[column].dtype=='object') | (len(train_df[column].unique())<=20) ]


# In[ ]:


numerical_var=train_df.columns.drop(categorical_var)


# In[ ]:


print("categorical_var length:", len(categorical_var), '\n'      "numerical_var length:", len(numerical_var))


# #2.Read target feature

# In[ ]:


train_df['target'].unique()


# In[ ]:


sns.countplot(train_df['target'])


# #3.Observe percentage null values
# 

# In[ ]:


null_values_columns=[]
percent_null_values=[]
for column in train_df.columns:
    percent_null_value=((train_df[column].isna().sum())/(len(train_df))) * 100
    if(percent_null_value > 0):
        null_values_columns.append(column)
        percent_null_values.append(percent_null_value)
df_null=pd.DataFrame(null_values_columns,columns=['column'])
df_null['percent_null_values']=percent_null_values
df_null.sort_values('percent_null_values',inplace=True)


plt.figure(figsize=(10,20))
plt.barh(df_null['column'],df_null['percent_null_values'])

for i, v in enumerate(df_null['percent_null_values']):
    plt.text(v, i, str(v), color='black', fontweight='bold')

plt.xlabel('percentage null values')
#     print(f'the percentage null values in column {column} is {((train_df[column].isna().sum())/(len(train_df))) * 100} %')


# In[ ]:


# making a new copy so that the original data set remains unchanged
df=train_df.copy()


# In[ ]:


df.shape


# In[ ]:


# removing id column
df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)


# #4.Handle case about day, month and year

# In[ ]:


#there are some date columns , lets convert them into different columns of day , month and year


# In[ ]:


def new_month_column(df2):
    return df2.month
def new_year_column(df2):
    return df2.year
def new_day_column(df2):
    return df2.day


# In[ ]:


def date_change(df1):
    df1[['feature_191','feature_192','feature_199','feature_201']]=df1[['feature_191','feature_192','feature_199','feature_201']].astype('datetime64')
    for column in ['feature_191','feature_192','feature_199','feature_201']:
        df1[column+'_day']=df1[column].apply(new_day_column)
        df1[column+'_month']=df1[column].apply(new_month_column)
        df1[column+'_year']=df1[column].apply(new_year_column)
    df1.drop(['feature_191','feature_192','feature_199','feature_201'],axis=1,inplace=True)
    return df1


# In[ ]:


date_change(df)
date_change(test_df)


# #5.Dropping columns with very large no, of null values (greater than 10%)

# In[ ]:


print("Null value length:", len(df_null))


# In[ ]:


large_null_columns=df_null[df_null['percent_null_values']>10]['column'].values


# In[ ]:


large_null_columns


# In[ ]:


df.drop(large_null_columns,axis=1,inplace=True)
test_df.drop(large_null_columns,axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


test_df.shape


# In[ ]:


null_values_columns=[]
percent_null_values=[]
for column in df.columns:
    percent_null_value=((df[column].isna().sum())/(len(df))) * 100
    if(percent_null_value > 0):
        null_values_columns.append(column)
        percent_null_values.append(percent_null_value)
df_null=pd.DataFrame(null_values_columns,columns=['column'])
df_null['percent_null_values']=percent_null_values
df_null.sort_values('percent_null_values',inplace=True)


plt.figure(figsize=(10,20))
plt.barh(df_null['column'],df_null['percent_null_values'])

for i, v in enumerate(df_null['percent_null_values']):
    plt.text(v, i, str(v), color='black', fontweight='bold')

plt.xlabel('percentage null values')


# In[ ]:


# successfully dropped the columns with large no. of na values


# #6.Deleting columns with high correlation

# In[ ]:


# deleting the columns with high correlation

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    return col_corr


# In[ ]:


high_corr_deleted=correlation(df,0.90)


# In[ ]:


test_df.drop(high_corr_deleted,axis=1,inplace=True)


# In[ ]:


print("high_corr_deleted length:", len(high_corr_deleted))


# In[ ]:


df.shape


# In[ ]:


df.head()


# #7.Deleting columns with zero variance (which means only one class)

# In[ ]:


# many columns have zero variance, we will delete these while data cleaning
def zero_var(df1):
    zero_var_columns=[]
    for column in df1.columns:
        if(len(df1[column].unique())==1):
            df1.drop(column, axis=1, inplace=True)
            zero_var_columns.append(column)
    return zero_var_columns

zero_var_columns=zero_var(df)


# In[ ]:


print("zero_var_columns length:", len(zero_var_columns))


# In[ ]:


test_df.drop(zero_var_columns,axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


test_df.shape


# In[ ]:


categorical_var=[column for column in df.columns if (df[column].dtype=='object')]
numerical_var=df.columns.drop(categorical_var)


# In[ ]:


df[categorical_var].head()


# In[ ]:


# lets manually examine some categorical variables


# In[ ]:


for column in categorical_var:
    print(column,' = ',len(df[column].unique()),'\t')


# #8.Handle categorical variables

# In[ ]:


# removing the categorical columns with more than 80 classes


# In[ ]:


def remove_more_than_80(df2):
    removed_columns_80=[]
    for column in categorical_var:
        if(len(df[column].unique())>70):
            df2.drop(column, axis=1, inplace=True)
#             categorical_var.remove(column)
            removed_columns_80.append(column)
    print('removed columns are:\n',removed_columns_80)
    return removed_columns_80


# In[ ]:


removed_columns_80=remove_more_than_80(df)


# In[ ]:


test_df.drop(removed_columns_80,axis=1,inplace=True)


# In[ ]:


print("categorical_var length:", categorical_var)


# In[ ]:


new_categorical_var = [feature for feature in categorical_var if feature not in removed_columns_80]


# In[ ]:


df.shape


# In[ ]:


test_df.shape


# In[ ]:


df.head()


# #9.Label encoding 
# 

# In[ ]:


def label_encoding(df1,df2):
    le=LabelEncoder()
    new=le.fit_transform(df1)
    
    df2.loc[~df2.isin(le.classes_)] = -1    
    df2.loc[df2.isin(le.classes_)] = le.transform(df2[df2.isin(le.classes_)])
   
    # new_test=le.transform(df2)
    return new,df2


# In[ ]:


# le_models=[]
for column in new_categorical_var:
    new,new_test=label_encoding(df[column],test_df[column])
    df[column]=new
    test_df[column]=new_test


# In[ ]:


df.head()


# In[ ]:


test_df.head()


# #10.Filling na values with -1

# In[ ]:


df.fillna(-1,inplace=True)
test_df.fillna(-1,inplace=True)


# In[ ]:


# X_train, X_test, y_train, y_test =train_test_split(df.drop('target',axis=1),df['target'],test_size=0.2,stratify=df['target'])


# In[ ]:


X_train=df.drop('target',axis=1)
y_train=df['target']


# In[ ]:


columns_with_na=[column for column in X_train.columns if X_train[column].isna().sum()>0]


# In[ ]:


columns_with_na


# In[ ]:


# no na values now


# In[ ]:


sns.countplot(df['target'])


# #11.Train the model with XGBoost Classifier

# In[ ]:


df.drop('target',axis=1).shape


# In[ ]:


X_train.shape


# In[ ]:


test_df.shape


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='auc')


# In[ ]:


xgboost_model.fit(X_train, y_train, verbose=False)


# In[ ]:


y_pred = xgboost_model.predict(test_df)


# In[ ]:


y_pred


# In[ ]:


len(y_pred)


# In[ ]:


id_list[0]


# In[ ]:


import csv


# In[ ]:


with open('submission.csv', 'w') as fp:
  writer = csv.writer(fp)
  writer.writerow(['id', 'target'])
  for i in range(len(y_pred)):
    writer.writerow([id_list[i], y_pred[i]])


# In[ ]:


X_train.shape

