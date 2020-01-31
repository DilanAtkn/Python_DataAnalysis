#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set() # setting seaborn default for plots
from sklearn.neighbors import *
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.naive_bayes import *
from sklearn.ensemble import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from prettytable import PrettyTable
import sys
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# # Load Dataset

# In[ ]:


df =pd.read_csv(r'C:\Users\dilan\OneDrive\Documents\incident_ predictor_backend\datepredictor\incident_event_log.csv')


# # Data Cleaning and Understanding

# In[ ]:


# removing extra information from feature values

df['caller_id'] = df['caller_id'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['opened_by'] = df['opened_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['closed_code'] = df['closed_code'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['resolved_by'] = df['resolved_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['sys_created_by'] = df['sys_created_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['sys_updated_by'] = df['sys_updated_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['location'] = df['location'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['category'] = df['category'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['subcategory'] = df['subcategory'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['u_symptom'] = df['u_symptom'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['cmdb_ci'] = df['cmdb_ci'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['impact'] = df['impact'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
df['urgency'] = df['urgency'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
df['priority'] = df['priority'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
df['assignment_group'] = df['assignment_group'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['assigned_to'] = df['assigned_to'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['problem_id'] = df['problem_id'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['vendor'] = df['vendor'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['caused_by'] = df['caused_by'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['closed_code'] = df['closed_code'].apply(lambda x: str(x).split(' ')[-1] if x != np.nan else np.nan)
df['opened_at'] = df['opened_at'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
df['resolved_at'] = df['resolved_at'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)
df['closed_at'] = df['closed_at'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)


# ## Missing values

# In[ ]:


df.replace('?', np.nan, inplace=True)


# In[ ]:


# Lets convert its datatype from object to int

df['reassignment_count'] = pd.to_numeric(df["reassignment_count"])
df['category'] = pd.to_numeric(df["category"])
df['subcategory'] = pd.to_numeric(df["subcategory"])
df['priority'] = pd.to_numeric(df["priority"])
df['assignment_group'] = pd.to_numeric(df["assignment_group"])


# In[ ]:


# Lets convert its datatype from object to datetime

df['opened_at']= pd.to_datetime(df['opened_at']) 
df['closed_at']= pd.to_datetime(df['closed_at']) 
df['sys_created_at']= pd.to_datetime(df['sys_created_at']) 
df['sys_updated_at']= pd.to_datetime(df['sys_updated_at']) 
df['resolved_at']= pd.to_datetime(df['resolved_at']) 


# In[ ]:


# subtraction of closing anf opening date of an incident return us days used for closing that incident
df['closed_at_opened_at'] = df['closed_at']-df['opened_at']


# In[ ]:


#Let's see instances with -ve values of closed_at_opened_at feature

e=df[df.closed_at_opened_at < '0']
e[['number', 'opened_at' , 'closed_at', 'closed_at_opened_at']].head()


# In[ ]:


p=e.index
df=df.drop(p)


# In[ ]:


df['incident_state'] = df['incident_state'].replace('-100', 'Active') 


# In[ ]:


q=df.incident_state.value_counts() # total count of values of incident_state are as follows


# In[ ]:


new_df = df.copy()


# In[ ]:


new_df.reset_index(inplace=True)


# In[ ]:


new_df.set_index('opened_at',inplace=True)
new_df.index = pd.to_datetime(new_df.index, utc=True)


# In[ ]:


m=new_df[new_df.incident_state == 'New']


# In[ ]:


monthly_resampled_opened_at_data = m.number.resample('M').count()


# ## Yearly
# 

# In[ ]:


yearly_resampled_opened_at_data  = m.number.resample('Y').count()


# ## Daily

# In[ ]:


daily_resampled_opened_at_data  = m.number.resample('D').count()


# # Let's dive in more depth

# In[ ]:


p=m.contact_type.value_counts().sort_values(ascending = False)


# In[ ]:


p=m.location.value_counts().sort_values(ascending = False)


# In[ ]:


df['location'] = df['location'].fillna('204')


# In[ ]:


p=m.category.value_counts().sort_values(ascending = False)


# In[ ]:


df['category'] = df['category'].fillna('53')


# In[ ]:


p=m.subcategory.value_counts().sort_values(ascending = False)


# In[ ]:


df['subcategory'] = df['subcategory'].fillna('174')


# In[ ]:


p=m.impact.value_counts().sort_values(ascending = False)


# In[ ]:


p=m.urgency.value_counts().sort_values(ascending = False)


# In[ ]:


p=m.priority.value_counts().sort_values(ascending = False)


# In[ ]:


p=m.notify.value_counts().sort_values(ascending = False)


# In[ ]:


selected_df= df.copy()


# In[ ]:


features_drop = [ 'number','active',  'made_sla', 'caller_id', 'opened_by','opened_at',
       'sys_created_by', 'sys_created_at', 'sys_updated_by', 'sys_updated_at', 'u_symptom', 
          'cmdb_ci', 'impact', 'urgency', 'assignment_group',
       'assigned_to', 'knowledge', 'u_priority_confirmation', 'notify',
       'problem_id', 'rfc', 'vendor', 'caused_by', 'closed_code',
       'resolved_by', 'resolved_at', 'closed_at', ]

selected_df = selected_df.drop(features_drop, axis=1) 


# # Covert all features into numeric 

# In[ ]:


selected_df['closed_at_opened_at'] = selected_df['closed_at_opened_at'].apply(lambda x: str(x).split(' ')[0] if x != np.nan else np.nan)


# In[ ]:


# Convert the data type of column 
selected_df = selected_df.astype({'category': 'int64', 'subcategory': 'int64', 'closed_at_opened_at': 'int64','location': 'int64'})


# # Label Encoding 

# In[ ]:


labelEncoder_incident_state = LabelEncoder()
labelEncoder_incident_state.fit(selected_df.incident_state)
selected_df['incident_state']=labelEncoder_incident_state.transform(selected_df.incident_state)


# In[ ]:


labelEncoder_contact_type = LabelEncoder()
labelEncoder_contact_type.fit(selected_df.contact_type)
selected_df['contact_type']=labelEncoder_contact_type.transform(selected_df.contact_type)


# In[ ]:


selected_df = selected_df.head(200)


# # Define training and testing set
# 

# In[ ]:


X= selected_df.drop('closed_at_opened_at', axis=1)
y= selected_df['closed_at_opened_at']
 


# In[ ]:


# split data into training and validation data, for both features and target 
# The split is based on a random number generator. Supplying a numeric value to 
# the random_state argument guarantees we get the same split every time we
# run this script. 
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) 


# # Preparing the Models.

# In[ ]:


def create_models(models):
    all_models = []

    for model in models:
        if model == "LogisticRegressionCV":
            all_models.append(('LogisticRegressionCV ', LogisticRegressionCV()))
        elif model == "KNeighborsClassifier":
            all_models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=3)))
        elif model == "SVC":
            all_models.append(('SVC', SVC()))
        elif model == "RandomForestClassifier":
            all_models.append(('RandomForestClassifier', RandomForestClassifier()))
         #   pickle.dump(model, open("RandomForestClassifier.pkl", "wb"))
        elif model == "DecisionTreeClassifier":
            all_models.append(('DecisionTreeClassifier', DecisionTreeClassifier() ))
        elif model == "GradientBoostingClassifier":
            all_models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
    return all_models


# # Training the Models.

# In[ ]:


def fit_models(all_models, train_X, val_X, train_y, val_y):
    
    t = PrettyTable(['Model','error'])
    print('')
    # Iterating all models one by one
    for name, model in all_models:
        print('Fitting:',name)
        trained_model = model.fit(train_X,train_y.values.ravel())
        prediction = trained_model.predict(val_X)
        error = mean_absolute_error(val_y, prediction)
       
        t.add_row([name, round(error,3)])
    print('\n\nDetailed Performance Of All Models.')
    print("=======================================")    
    print(t)


# # Main

# In[ ]:


models = ["LogisticRegressionCV","KNeighborsClassifier","SVC","RandomForestClassifier","DecisionTreeClassifier","GradientBoostingClassifier"]

# Create Models
all_models = create_models(models)

# Model Evaluation
fit_models(all_models, train_X, val_X, train_y, val_y)


# In[ ]:


BestModel=DecisionTreeClassifier()


# #  Train Best Model on All Data

# In[ ]:


x_alldata= selected_df.drop('closed_at_opened_at', axis=1)
y_alldata= selected_df['closed_at_opened_at']


# In[ ]:


BestModel.fit(x_alldata, y_alldata)


# # Save the Trained Model as Pickle File
# 

# In[ ]:


file = open(r'C:\Users\dilan\OneDrive\Documents\incident_ predictor_backend\datepredictor\app\BestModel.pkl','wb')
pickle.dump(BestModel,file)
file.close()


# # Load the Trained Model 

# In[ ]:


# import pickle
infile = open(r'C:\Users\dilan\OneDrive\Documents\incident_ predictor_backend\datepredictor\app\BestModel.pkl','rb')
bestmodel = pickle.load(infile)
infile.close()


#  # Take Input from User

# In[ ]:


incident_state=input("Please Enter incident_state here: ")
reassignment_count=input("Please Enter your reassignment_count: ")
reopen_count  =input("Please Enter reopen_count: ")
sys_mod_count = input("Please Enter sys_mod_count: ")
contact_type = input("Please Enter contact_type: ")
location  =input("Please Enter location: ")
category  =input("Please Enter category: ")
subcategory  =input("Please Enter subcategory: ")
priority  =input("Please Enter priority: ")


# In[ ]:


dictt={'incident_state':[incident_state],'reassignment_count':[reassignment_count], 'reopen_count': [reopen_count],'sys_mod_count':[sys_mod_count ],'contact_type':[contact_type],'location':[location],'category':[category] 
       ,'subcategory':[subcategory],'priority':[priority]}
    


# # Convert User Input into Feature Vector 

# In[ ]:


input1 = pd.DataFrame(dictt)
print("User Input in Actual Dataframe form : ")


# In[ ]:


labelEncoder_incident_state.transform(input1.incident_state)
labelEncoder_contact_type.transform(input1.contact_type)


# In[ ]:



input1['incident_state'] =labelEncoder_incident_state.transform(input1.incident_state)
input1['contact_type'] =labelEncoder_contact_type.transform(input1.contact_type)


# In[ ]:


print("User Input in Encoded Dataframe form : ")


# # Apply Trained Model on Feature Vector of  Unseen Data 

# In[ ]:


y_predict = bestmodel.predict(input1)


# #  Output Prediction to User

# In[ ]:


print("Prediction : {} days are required to close this incident ".format((y_predict)))

