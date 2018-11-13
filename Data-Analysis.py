
# coding: utf-8

# ## Getting the data

# http://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html

# ## Cleaning our data

# In[1]:


import pandas as pd


# In[2]:


import csv
from collections import defaultdict

disease_list = []

def return_list(disease):
    disease_list = []
    match = disease.replace('^','_').split('_')
    ctr = 1
    for group in match:
        if ctr%2==0:
            disease_list.append(group)
        ctr = ctr + 1

    return disease_list

with open("Scraped-Data/dataset_uncleaned.csv") as csvfile:
    reader = csv.reader(csvfile)
    disease=""
    weight = 0
    disease_list = []
    dict_wt = {}
    dict_=defaultdict(list)
    for row in reader:

        if row[0]!="\xc2\xa0" and row[0]!="":
            disease = row[0]
            disease_list = return_list(disease)
            weight = row[1]

        if row[2]!="\xc2\xa0" and row[2]!="":
            symptom_list = return_list(row[2])

            for d in disease_list:
                for s in symptom_list:
                    dict_[d].append(s)
                dict_wt[d] = weight

    #print (dict_)


# Writing our cleaned data

# In[3]:


with open("Scraped-Data/dataset_clean.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    for key,values in dict_.items():
        for v in values:
            #key = str.encode(key)
            key = str.encode(key).decode('utf-8')
            #.strip()
            #v = v.encode('utf-8').strip()
            #v = str.encode(v)
            writer.writerow([key,v,dict_wt[key]])


# In[4]:


columns = ['Source','Target','Weight']


# In[5]:


data = pd.read_csv("Scraped-Data/dataset_clean.csv",names=columns, encoding ="ISO-8859-1")


# In[6]:


data.head()


# In[7]:


data.to_csv("Scraped-Data/dataset_clean.csv",index=False)


# In[8]:


slist = []
dlist = []
with open("Scraped-Data/nodetable.csv","w") as csvfile:
    writer = csv.writer(csvfile)

    for key,values in dict_.items():
        for v in values:
            if v not in slist:
                writer.writerow([v,v,"symptom"])
                slist.append(v)
        if key not in dlist:
            writer.writerow([key,key,"disease"])
            dlist.append(key)


# In[9]:


nt_columns = ['Id','Label','Attribute']


# In[10]:


nt_data = pd.read_csv("Scraped-Data/nodetable.csv",names=nt_columns, encoding ="ISO-8859-1",)


# In[11]:


nt_data.head()


# In[12]:


nt_data.to_csv("Scraped-Data/nodetable.csv",index=False)


# ## Analysing our cleaned data

# In[13]:


data = pd.read_csv("Scraped-Data/dataset_clean.csv", encoding ="ISO-8859-1")


# In[14]:


data.head()


# In[15]:


df = pd.DataFrame(data)


# In[16]:


df_1 = pd.get_dummies(df.Target)


# In[17]:


df_1.head()


# In[18]:


df.head()


# In[19]:


df_s = df['Source']


# In[20]:


df_pivoted = pd.concat([df_s,df_1], axis=1)


# In[21]:


df_pivoted.drop_duplicates(keep='first',inplace=True)


# In[22]:


df_pivoted[:5]


# In[23]:


cols = df_pivoted.columns


# In[24]:


cols = cols[1:]


# In[25]:


df_pivoted = df_pivoted.groupby('Source').sum()
df_pivoted = df_pivoted.reset_index()
df_pivoted[:5]


# In[26]:


df_pivoted.to_csv("Scraped-Data/df_pivoted.csv")


# In[27]:


x = df_pivoted[cols]
y = df_pivoted['Source']


# ### Trying out our classifier to learn diseases from the symptoms

# In[28]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[30]:


mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)


# In[31]:


mnb.score(x_test, y_test)


# In[32]:


mnb_tot = MultinomialNB()
mnb_tot = mnb_tot.fit(x, y)


# In[33]:


mnb_tot.score(x, y)


# In[34]:


disease_pred = mnb_tot.predict(x)


# In[35]:


disease_real = y.values


# In[36]:


for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0} Actual:{1}'.format(disease_pred[i], disease_real[i]))


# ### Training a decision tree

# In[37]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[38]:


print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x,y)
print ("Acurracy: ", clf_dt.score(x,y))


# In[39]:


from sklearn import tree 
from sklearn.tree import export_graphviz

export_graphviz(dt, 
                out_file='DOT-files/tree.dot', 
                feature_names=cols)


# ## Analysis of the Manual data

# In[40]:


data = pd.read_csv("Manual-Data/Training.csv")


# In[41]:


data.head()


# In[42]:


data.columns


# In[43]:


df = pd.DataFrame(data)


# In[44]:


df.head()


# In[45]:


cols = df.columns


# In[46]:


cols = cols[:-1]


# In[47]:


cols


# In[48]:


x = df[cols]
y = df['prognosis']


# ### Trying out our classifier to learn diseases from the symptoms

# In[49]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[51]:


mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)


# In[52]:


mnb.score(x_test, y_test)


# In[53]:


from sklearn import cross_validation
print ("cross result========")
scores = cross_validation.cross_val_score(mnb, x_test, y_test, cv=3)
print (scores)
print (scores.mean())


# We use the testing dataset to actually test our Multinomial Bayes model

# In[54]:


test_data = pd.read_csv("Manual-Data/Testing.csv")


# In[55]:


test_data.head()


# In[56]:


testx = test_data[cols]
testy = test_data['prognosis']


# In[57]:


mnb.score(testx, testy)


# ### Training a decision tree

# In[58]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[59]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[60]:


print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)
print ("Acurracy: ", clf_dt.score(x_test,y_test))


# In[61]:


from sklearn import cross_validation
print ("cross result========")
scores = cross_validation.cross_val_score(dt, x_test, y_test, cv=3)
print (scores)
print (scores.mean())


# In[62]:


print ("Acurracy on the actual test data: ", clf_dt.score(testx,testy))


# In[63]:


from sklearn import tree 
from sklearn.tree import export_graphviz

export_graphviz(dt, 
                out_file='DOT-files/tree.dot', 
                feature_names=cols)


# In[64]:


dt.__getstate__()


# #### Finding the Feature importances

# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")


# In[66]:


features = cols


# In[78]:


for f in range(10):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))


# In[68]:


export_graphviz(dt, 
                out_file='DOT-files/tree-top5.dot', 
                feature_names=cols,
                max_depth = 5
               )


# In[69]:


feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i


# In[70]:



x = []
x.append(feature_dict['extra_marital_contacts'])
x.append(feature_dict['shivering'])
x.append(feature_dict['joint_pain'])
x.append(feature_dict['vomiting'])
print(x)


# In[71]:


sample_x=[]
for i in range(len(features)):
    if i in x:
        sample_x.append(1.0)
    else:
        sample_x.append(0.0)
print(sample_x)


# In[72]:


len(sample_x)


# In[73]:


sample_x = np.array(sample_x).reshape(1,len(sample_x))
print(sample_x)


# In[74]:


dt.predict(sample_x)


# In[75]:


dt.predict_proba(sample_x)

