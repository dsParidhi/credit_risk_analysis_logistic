
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# In[54]:


train_input = pd.read_csv("D:\pythonfiles\Python\Train.csv")


# In[55]:


test_input = pd.read_csv("D:\pythonfiles\Python\Validate.csv")


# In[56]:


train_input.info()


# In[57]:


train_input.columns


# In[58]:


test_input.columns


# In[59]:


#the last column has a different name in both
#lets make the names same. and then merge them togehter
#so that we can fill the missing values simulteneously
test_input.rename(columns={"outcome": "Loan_Status"},inplace=True)


# In[60]:


all=pd.concat([train_input,test_input],axis=0)
all.shape


# In[61]:


print(all.tail())
all.reset_index(inplace=True,drop=True)# reset index else merging will have
#issues


# In[62]:


all.isnull().sum()#gives the missing value of all columns


# In[63]:


all.shape # read the decription of each column from the word document


# In[72]:


# before proceeding to Model Building, lets fill the missing values
Counter(all['Gender'])


# In[73]:


#lets fill them by Male
print(all[all['Gender'].isnull()].index.tolist())
#these rows are null for gender
#lets fill them with the Model of Gender  i.e Male
gender_null=all[all['Gender'].isnull()].index.tolist()


# In[74]:


all['Gender'].iloc[gender_null]="Male"


# In[78]:


#check if filed
print(sum(all['Gender'].isnull()))#oky done
Counter(all['Gender'])


# In[79]:


#lets fill Married now
print(Counter(all['Married']))#most are married
#lets fill them Yes if they have dependents else No
pd.crosstab(all['Married'].isnull(),all['Dependents'].isnull())
# for all 3 missing values in Married , the # of dependents are also missing
# let fill them with the Yes--i.e married as most customers are marrried


# In[80]:


married_null=all[all['Married'].isnull()].index.tolist()
married_null


# In[81]:


all['Married'].iloc[married_null]="Yes"


# In[82]:


all.isnull().sum()


# In[83]:


Counter(all['Dependents'])


# In[84]:


# elts see the Dependents wrt Marriage
pd.crosstab(all['Married'],all['Dependents'].isnull())


# In[85]:


pd.crosstab(all['Dependents'],all['Married'])


# In[86]:


# for the bacheors, lets fill the missing dependents as 0
#lets find the index of all rows with Depednednts mssing and Married NO
bachelor_nulldependent=all[(all['Married']=="No") & (all['Dependents'].isnull())].index.tolist()
print(bachelor_nulldependent)


# In[87]:


all['Dependents'].iloc[bachelor_nulldependent]='0'


# In[88]:


Counter(all['Dependents'])


# In[89]:


#for the remaining 16 missing depemdents,
#let see how mnay dependents Male & Female have
pd.crosstab(all['Gender'],all['Dependents'])


# In[90]:


# so feamle have less dependents
#lets see the gender of the 8 missing dependents
all['Gender'].iloc[all[all['Dependents'].isnull()].index.tolist()]


# In[91]:


# all of them are Male
# lets fill them with the mode of all dependent of male
pd.crosstab(all['Gender'],all['Dependents'])


# In[92]:


#let sfill the # dependent with 1
all['Dependents'].iloc[all[all['Dependents'].isnull()].index.tolist()]="0"


# In[93]:


all.isnull().sum()


# In[94]:


Counter(all['Self_Employed'])


# In[95]:


self_emp_null=all[all['Self_Employed'].isnull()].index.tolist()


# In[96]:


#fill missing selfemployed with NO
all['Self_Employed'].iloc[self_emp_null]="No"


# In[97]:


all.isnull().sum()


# In[98]:


pd.crosstab(all['LoanAmount'].isnull(),all['Loan_Amount_Term'])


# In[99]:


all.groupby(all['Loan_Amount_Term'])['LoanAmount'].mean()


# In[100]:


#lets fill the missing values in LoanAmount 
#with the mean of the respective Loan_Term
#we see that 180 & 240 has the almost same Loan amount 128-131
#& 360 has high i.e 144
#so lets fill only 360 by 144 
#and all remaining by 130
all['LoanAmount'][(all['LoanAmount'].isnull()) & (all['Loan_Amount_Term']==360)]=144


# In[101]:


all['LoanAmount'][(all['LoanAmount'].isnull())]=130


# In[102]:


#lets fill Loan Amount Term
(all['Loan_Amount_Term']).value_counts()


# In[103]:


#lets find the Loan Tenure by the mode i,e 512
all['Loan_Amount_Term'][all['Loan_Amount_Term'].isnull()]=360


# In[104]:


all.isnull().sum()


# In[105]:


all['Credit_History'].value_counts()


# In[106]:


pd.crosstab(all['Gender'],all['Credit_History']) 
# Gender makes no difference


# In[107]:


pd.crosstab(all['Self_Employed'],all['Credit_History'])
# Self_Employed makes no difference


# In[108]:


pd.crosstab(all['Education'],all['Credit_History'])
# Education makes no difference


# In[109]:


pd.crosstab(all['Married'],all['Credit_History'])
# married makes no difference


# In[110]:


# run a logistic regression to fill the Credit History for the 50 missing values


# In[111]:


all.isnull().sum()


# In[112]:


#prepare a train set which has all values of Credit History present
#make the rows having Credit History missig as test set
#before spliting lets first create the dummies
all.columns
#df_dummy1=#for Gender
#df_dummy2=#for married
#df_dummy3=#for dependents
#df_dummy4=
#convert one columns at a time by using pd.get_dummies or use
all_new=pd.get_dummies(all.drop(['Loan_ID'],axis=1),drop_first=True)


# In[113]:


all_new.head()


# In[114]:


all_new.isnull().sum()


# In[115]:


#split into train test
test=all_new[all_new['Credit_History'].isnull()]
all_in_test=test.index.tolist()
test.head()


# In[116]:


all_in_train=[x for x in all_new.index.tolist() if x not in all_in_test]
train=all_new.iloc[all_in_train]
train.shape


# In[117]:


X_train=train.drop(['Loan_Status_Y','Credit_History'],axis=1)
Y_train=train['Credit_History']


# In[118]:


print(X_train.columns)


# In[119]:


X_test=test.drop(['Loan_Status_Y','Credit_History'],axis=1)
Y_test=test['Credit_History']


# In[120]:


#logistic regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,Y_train)


# In[121]:


print(X_test.head())


# In[122]:


pred=log_reg.predict(X_test)


# In[123]:


print(pred)


# In[124]:


#replace the Nan's in the test with these values
test['Credit_History']=pred
print(pred)


# In[125]:


#remerge the train & test
df_all=pd.concat([train,test],axis=0)


# In[126]:


df_all.shape


# In[127]:


df_all.head()
df_all.isnull().sum()
#all good


# In[128]:


train2=df_all.head(len(train_input))
test2=df_all.tail(len(test_input))


# In[129]:


X_train=train2.drop(['Loan_Status_Y'],axis=1)
Y_train=train2['Loan_Status_Y']
X_test=test2.drop(['Loan_Status_Y'],axis=1)
Y_test=test2['Loan_Status_Y']


# In[130]:


print(X_test.head())

