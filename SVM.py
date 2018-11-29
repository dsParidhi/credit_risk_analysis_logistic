
# coding: utf-8

# In[14]:



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# In[15]:


train_input=pd.read_csv("D:\pythonfiles\Python\Train.csv")
test_input=pd.read_csv("D:\pythonfiles\Python\Validate.csv")


# In[16]:


print(train_input.columns)
print(test_input.columns)


# In[17]:


#the last column has a different name in both,
#lets make the names same. and then merge them togehter
#so that we can fill the missing values simulteneously
test_input.rename(columns={"outcome": "Loan_Status"},inplace=True)


# In[18]:


all=pd.concat([train_input,test_input],axis=0)
all.shape


# In[19]:


print(all.tail())
all.reset_index(inplace=True,drop=True)# reset index else merging will have
#issues, 


# In[20]:


all.isnull().sum()#gives the missing value of all columns


# In[21]:


all.shape # read the decription of each column from the word document


# In[22]:


# before proceeding to Model Building, lets fill the missing values


# In[23]:


Counter(all['Gender'])


# In[24]:


#lets fill them by Male
print(all[all['Gender'].isnull()].index.tolist())
#these rows are null for gender
#lets fill them with the Model of Gender  i.e Male
gender_null=all[all['Gender'].isnull()].index.tolist()


# In[25]:


all['Gender'].iloc[gender_null]="Male"


# In[27]:


#check if filed
print(sum(all['Gender'].isnull()))#oky done
Counter(all['Gender'])


# In[29]:


#check if filed
print(sum(all['Gender'].isnull()))#oky done
Counter(all['Gender'])


# In[31]:


#lets fill Married now
print(Counter(all['Married']))#most are married
#lets fill them Yes if they have dependents else No
pd.crosstab(all['Married'].isnull(),all['Dependents'].isnull())
# for all 3 missing values in Married , the # of dependents are also missing
# let fill them with the Yes--i.e married as most customers are marrried


# In[32]:


married_null=all[all['Married'].isnull()].index.tolist()
married_null


# In[33]:


all['Married'].iloc[married_null]="Yes"


# In[34]:


all.isnull().sum()


# In[35]:


Counter(all['Dependents'])


# In[36]:


# elts see the Dependents wrt Marriage
pd.crosstab(all['Married'],all['Dependents'].isnull())


# In[37]:


pd.crosstab(all['Dependents'],all['Married'])


# In[40]:


# for the bacheors, lets fill the missing dependents as 0
#lets find the index of all rows with Depednednts mssing and Married NO
bachelor_nulldependent=all[(all['Married']=="No") & (all['Dependents'].isnull())].index.tolist()
print(bachelor_nulldependent)


# In[41]:


all['Dependents'].iloc[bachelor_nulldependent]='0'


# In[42]:


Counter(all['Dependents'])


# In[43]:


#for the remaining 16 missing depemdents,
#let see how mnay dependents Male & Female have
pd.crosstab(all['Gender'],all['Dependents'])


# In[44]:


# so feamle have less dependents
#lets see the gender of the 8 missing dependents
all['Gender'].iloc[all[all['Dependents'].isnull()].index.tolist()]


# In[45]:


# all of them are Male
# lets fill them with the mode of all dependent of male
pd.crosstab(all['Gender'],all['Dependents'])


# In[46]:


#let sfill the # dependent with 1
all['Dependents'].iloc[all[all['Dependents'].isnull()].index.tolist()]="0"


# In[47]:


all.isnull().sum()


# In[48]:


Counter(all['Self_Employed'])


# In[49]:


self_emp_null=all[all['Self_Employed'].isnull()].index.tolist()


# In[50]:


#fill missing selfemployed with NO
all['Self_Employed'].iloc[self_emp_null]="No"


# In[51]:


all.isnull().sum()


# In[52]:


pd.crosstab(all['LoanAmount'].isnull(),all['Loan_Amount_Term'])


# In[53]:


all.groupby(all['Loan_Amount_Term'])['LoanAmount'].mean()


# In[54]:


#lets fill the missing values in LoanAmount 
#with the mean of the respective Loan_Term
#we see that 180 & 240 has the almost same Loan amount 128-131
#& 360 has high i.e 144
#so lets fill only 360 by 144 
#and all remaining by 130
all['LoanAmount'][(all['LoanAmount'].isnull()) & (all['Loan_Amount_Term']==360)]=144


# In[55]:


all['LoanAmount'][(all['LoanAmount'].isnull())]=130


# In[56]:


#lets fill Loan Amount Term
(all['Loan_Amount_Term']).value_counts()


# In[57]:


#lets find the Loan Tenure by the mode i,e 512
all['Loan_Amount_Term'][all['Loan_Amount_Term'].isnull()]=360


# In[58]:


all.isnull().sum()


# In[59]:


all['Credit_History'].value_counts()


# In[60]:


pd.crosstab(all['Gender'],all['Credit_History']) 
# Gender makes no difference


# In[61]:


pd.crosstab(all['Self_Employed'],all['Credit_History'])
# Self_Employed makes no difference


# In[62]:


pd.crosstab(all['Education'],all['Credit_History'])
# Education makes no difference


# In[63]:


pd.crosstab(all['Married'],all['Credit_History'])
# married makes no difference


# In[64]:


# run a logistic regression to fill the Credit History for the 50 missing values


# In[65]:


all.isnull().sum()


# In[66]:


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


# In[67]:


all_new.head()


# In[68]:


all_new.isnull().sum()


# In[69]:


#split into train test
test=all_new[all_new['Credit_History'].isnull()]
all_in_test=test.index.tolist()
test.head()


# In[70]:


all_in_train=[x for x in all_new.index.tolist() if x not in all_in_test]
train=all_new.iloc[all_in_train]
train.shape


# In[71]:


X_train=train.drop(['Loan_Status_Y','Credit_History'],axis=1)
Y_train=train['Credit_History']


# In[91]:


X_test=test.drop(['Loan_Status_Y','Credit_History'],axis=1)
Y_test=test['Credit_History']


# # SVM in Python

# In[92]:


from sklearn.svm import SVC
model=SVC()
model.fit(X_train,Y_train)


# In[93]:


#above the default values of C & gamma is taken
pred=model.predict(X_test)

print(Y_test)


# In[94]:


from sklearn.metrics import classification_report,confusion_matrix


# In[95]:


classification_report(Y_test,pred)


# In[96]:


print(confusion_matrix(Y_test,pred))


# In[101]:


print(Counter(test['Loan_Status_Y']))
baselin_acccuracy=float(282)/(282+85)
print(baselin_acccuracy)
Accuracy=float(282+1)/(282+1+84)
Accuracy # not very good compared to basline


# In[102]:


#lets play with the C & gamma parameter in SVM
from sklearn.grid_search import GridSearchCV


# In[103]:


param_grid={'C':[0.001,0.01],
           'gamma':[1,0.08,0.09],
           'kernel': ['linear']}


# In[104]:


grid=GridSearchCV(SVC(),param_grid,verbose=2)


# In[105]:


grid.fit(X_train,Y_train)


# In[ ]:


print(grid.best_params_)
print(grid.best_estimator_)


# In[ ]:


grid_pred=grid.predict(X_test)


# In[ ]:


Counter(grid_pred)


# In[ ]:


print(confusion_matrix(Y_test,grid_pred))

