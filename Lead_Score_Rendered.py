#!/usr/bin/env python
# coding: utf-8

# 
# LOGISTIC REGRESSION MODEL
# 
# Lead scoring Case Study
# 
# Problem Statement: In this notebook, we will build a binary classification model to predict potential leads dataset by calculating lead score which will accelerate the company's revenue.
# Essentially, X Education Company wants to know potential customers who are likely to be converted into hot leads.
# 
# Based on survey they got Lead scoring dataset, converted is target variable with binary data
# 
# The equation found on Logistic Regression Model, converted=-1.884+(1.177TotalVisits')+(1.177'Total Time Spent on Website')+(4.69'Lead Source_Olark Chat')+(1.566 'Lead Source_Reference')+(4.111'Lead Source_Welingak Website')+(5.821'Last Activity_Olark Chat Conversation')+(-0.869'Last Activity_Others')+(-0.722'Specialization_Travel and Tourism')+(2.096'What is your current occupation_Working Professional')+(-1.748'What matters most to you in choosing a course_Other')+(-4.570'Tags_Ringing')+(1.864'Last Notable Activity_SMS Sent'))
# 

# In[1]:


# Importing necessory libraries
import numpy as np
import pandas as pd
#import the warnings.
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#Model Building libraries
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[277]:


# Readthe Data
lead_df=pd.read_csv("Leads.csv")


# In[278]:


# Check the head of the dataset
lead_df.head()


# In[279]:


lead_df.shape


# In[280]:


lead_df.info()


# # Data Cleaning

# In[281]:


#Check duplicate rows in a DataFrame
lead_df.duplicated().any()


# In[282]:


percent=round(lead_df.isnull().sum()/len(lead_df)*100,2).sort_values(ascending=False)
total=lead_df.isnull().sum().sort_values(ascending=False)
pd.concat([percent,total],axis=1,keys=['Percent','Total'])


# In[283]:


#Handle Select value present in many categorical Variable by replacing 'Select' with NaN
lead_df.replace({'Select': np.nan}, inplace = True)


# In[284]:


# Checking the missing percenetage
percent=round(lead_df.isnull().sum()/len(lead_df)*100,2).sort_values(ascending=False)
total=lead_df.isnull().sum().sort_values(ascending=False)
pd.concat([percent,total],axis=1,keys=['Percent','Total'])


# In[285]:


#Removing columns  Lead Number,Prospect ID  as it doesnt add much value for further analysis
lead_df.drop(['Prospect ID','Lead Number'], axis = 1, inplace = True)


# In[286]:


lead_df.shape


# In[287]:


# #Deleting 30% missing columns at once by passing the thresh to drop na
missing_data_col=len(lead_df)*0.6
df=lead_df.dropna(thresh=missing_data_col,axis=1)


# In[288]:


df.shape


# In[289]:


# Display coumns with only missing values along with their percentage
x=(df.isnull().sum()/df.shape[0]*100)
x[x > 0].sort_values(ascending = False)


# In[290]:


# Get the value counts of all the columns

for column in df:
    print(df[column].astype('category').value_counts())

    print('***********************************************************')


# In[46]:


# check data skewness  If the mean is greater than the median, then the distribution is said to be positively skewed. If the mean is less than the median, then the distribution is said to be negatively skewed.
df.describe()


# In[47]:


df.describe(include='all')


# In[307]:


print("Values by absolute number\n",df['Converted'].value_counts())
print("Values by %\n",df['Converted'].value_counts(normalize=True) * 100)
# s = df['Converted'].value_counts(normalize=True) * 100
# s.plot.bar();

sns.countplot(x = df['Converted'], hue = df['Converted'], palette = 'deep');


# In[48]:


def hue_count(x, y, p = 'deep'):
    ax = sns.countplot(x = df[x], hue = df[y], palette = p)
    ax.set_title('"{}" Composition'.format(x), fontsize = 15, fontweight = 'bold', pad = 5)
    ax.set_xlabel(x, fontsize = 14)
    ax.set_ylabel('Count', fontsize = 14)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1.5)


# In[58]:


# Conversion Rate = (Number of Conversions / Total number of Leads) * 100
def conversion_rate_plot(x, y = 'Converted', p = 'deep'):
    ax = sns.barplot(x = df[x], y = df[y], palette = p)
    ax.set_title('Conversion Rate'.format(y,x), fontsize = 15, fontweight = 'bold', pad = 5)
    ax.set_xlabel(x, fontsize = 14)
    ax.set_ylabel('Rate', fontsize = 14)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1.5)
    for k in ax.patches:
        ax.annotate('{:.1f} %'.format(k.get_height()*100),(k.get_x()+0.25, k.get_height()))


# Handling missing values

# Inspect Categoical column
# 

# In[50]:



print("null vales",100*df.City.isnull().sum()/len(df.City))


# In[51]:


df['City_new']=np.where(df['City'].isin(['Mumbai','Thane & Outskirts']),df['City'],'Others Metro Cities')


# In[52]:


print(100*df.City.value_counts(normalize=True))
df.drop('City',axis=1,inplace=True)


# In[59]:


# Count plot
plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count(x = 'City_new', y = 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('City_new')
plt.xticks(rotation = 45)

plt.tight_layout()


# Specialization

# In[60]:


print("Null values",100*df.Specialization.isnull().sum()/len(df.Specialization))
print("------------------------")
print(100*df.Specialization.value_counts(normalize=True))
print("------------------------")
print(df.Specialization.mode()[0])
df.Specialization.fillna("Others",inplace=True)




# In[61]:


#combining less frequent levels into one, 'Others'
x = 100*df['Specialization'].value_counts(normalize = True)
df['Specialization'] = df['Specialization'].replace(list(x[x < 2].index), 'Others')

df['Specialization'].value_counts(normalize = True).mul(100).round(2)


# In[62]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Specialization', 'Converted')
plt.xticks(rotation = 45)
# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Specialization', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# In[ ]:





# Analysis:Leads from Management sector, like Business administration,operation mgmt, HR and Marketing Management, and Banking, Investment and Insurance specialization,supply chain mgmt,finance mgmt are relatively more likely to convert. Their average conversion rate is higher than the overall average. These groups can be targettd more.

# Tags

# In[63]:


print(100*df.Tags.value_counts(normalize=True))
print("--------------------------------")
print("null values", df.Tags.isnull().sum())


# In[64]:


# fill the missing values with mode value of Tags.
df.Tags.fillna(df.Tags.mode()[0],inplace=True)
100*df.Tags.value_counts(normalize=True)


# In[65]:


#combining less frequent levels into one, 'Others'
x = 100*df['Tags'].value_counts(normalize = True)
df['Tags'] = df['Tags'].replace(list(x[x < 6].index), 'Not intersted')


# In[66]:


100*df.Tags.value_counts(normalize=True)


# In[67]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Tags', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Tags', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# Analysis:Customer with current status "Will revert after reading the email" have moderatly high chance of converting compared to others falling in this group

# What matters most to you in choosing a course

# In[68]:


print("null values",df['What matters most to you in choosing a course'].isnull().sum()/len(df['What matters most to you in choosing a course']))
print(100*df['What matters most to you in choosing a course'].value_counts(normalize=True))


# In[69]:


df['What matters most to you in choosing a course'].fillna("Other",inplace=True)
df['What matters most to you in choosing a course'].replace(to_replace="Flexibility & Convenience", value='Better Career Prospects', inplace=True)


# In[70]:


df['What matters most to you in choosing a course'].value_counts()


# In[71]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('What matters most to you in choosing a course', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('What matters most to you in choosing a course', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# Analysis:customer aspiring "Better career Prospect"  are likely to be converted.Should target this type of customers

# What is your current occupation

# In[72]:


print("null values",df['What is your current occupation'].isnull().sum()/len(df['What is your current occupation']))
print("-----------------------------------------------")
print(100*df['What is your current occupation'].value_counts(normalize=True))


# In[73]:


#imputing culumn with mode, 'Unemployed'
df['What is your current occupation'].fillna('Unemployed', inplace = True)


# In[74]:


x = 100*df['What is your current occupation'].value_counts(normalize = True)
df['What is your current occupation'] =df['What is your current occupation'].replace(list(x[x < 4].index), 'Other')
print(100*df['What is your current occupation'].value_counts(normalize=True))


# In[75]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('What is your current occupation', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('What is your current occupation', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# *Analysis : 1.Working professionals occupation should be targeted since their  conversion rate is  92%

# Country

# In[76]:


print(100*df.Country.value_counts(normalize=True,dropna=False) )
print("-----------------------------------------")
print("Null values",df.Country.isnull().sum())


# In[77]:


# Imputting culumns
df.Country.fillna("Missing",inplace=True)
df['Country_new']=np.where(df['Country'].isin(['India']),df['Country'],'Others')


# In[78]:


df['Country_new'].value_counts() /len(df.Country)


# In[79]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Country_new', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Country_new', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)
plt.tight_layout()


# In[80]:


df.drop(['Country'] , axis= 1,inplace=True)


# Lead Source

# In[81]:


# Imputting missing values and values with <55 to others and chang google to Google in Lead Source
print(100*df['Lead Source'].value_counts(normalize=True))
print("null values :",df['Lead Source'].isnull().sum())


# In[82]:


# Drop null values in the 'Lead Source' column
df = df.dropna(subset=['Lead Source'])
print("null values :",df['Lead Source'].isnull().sum())


# In[83]:



df["Lead Source"].replace(to_replace="google", value=df['Lead Source'].mode()[0], inplace=True)
df["Lead Source"].replace(to_replace="Facebook", value='others', inplace=True)
df["Lead Source"].replace(to_replace="bing", value='others' , inplace=True)
df["Lead Source"].replace(to_replace="Click2call", value='others' , inplace=True)
df["Lead Source"].replace(to_replace="Press_Release", value='others' , inplace=True)
df["Lead Source"].replace(to_replace="Social Media", value='others' , inplace=True)
df["Lead Source"].replace(to_replace="Live Chat", value='others' , inplace=True)
df["Lead Source"].replace(to_replace="youtubechannel", value='others', inplace=True)
df["Lead Source"].replace(to_replace="testone", value='others', inplace=True)
df["Lead Source"].replace(to_replace="Pay per Click Ads", value='others', inplace=True)
df["Lead Source"].replace(to_replace="welearnblog_Home", value='others', inplace=True)
df["Lead Source"].replace(to_replace="WeLearn", value='others', inplace=True)
df["Lead Source"].replace(to_replace="blog", value='others', inplace=True)
df["Lead Source"].replace(to_replace="Referral Sites", value='others', inplace=True)
df["Lead Source"].replace(to_replace="NC_EDM", value='others', inplace=True)

# df["Lead Source"].replace(to_replace="Reference", value='others', inplace=True)
# df["Lead Source"].replace(to_replace="Welingak Website", value='others', inplace=True)


# In[84]:


print(df['Lead Source'].value_counts(normalize=True))
# print("null values :",df['Lead Source'].isnull().sum())


# In[88]:


df['Lead Source'].value_counts(normalize=True).mul(100).plot(kind = 'bar', figsize = [10,5])
plt.title(" Lead Source Categories");


#  *Analysis:
#  1.Google is a major source of leads when bringing the clients but they are less likely to be converted compared to customers through "Welingak Website" and "Reference " .
#  2."Welingak Website should be our top priority since they are more likely to be converted.

# In[89]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Lead Source', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Lead Source', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# B.Inspect Numeric data 1:

# TotalVisits

# In[90]:


print("NullValues:",df['TotalVisits'].isnull().sum()/len(df['TotalVisits']))
print(df['TotalVisits'].fillna(df['TotalVisits'].median(), inplace = True))


# In[91]:


sns.boxplot(df.TotalVisits)
plt.show()


# In[92]:


df.TotalVisits.quantile([0.6,0.75,0.95,0.99,1])


# In[93]:


Q11 = df['TotalVisits'].quantile(0.00)
Q14 = df['TotalVisits'].quantile(0.99)
df['TotalVisits'][df['TotalVisits'] <= Q11]=Q11
df['TotalVisits'][df['TotalVisits'] >= Q14]=Q14


# In[94]:


sns.boxplot(df.TotalVisits)
plt.show()


# In[95]:


df['TotalVisits'].value_counts(normalize=True)


# In[96]:



# Create a Pandas Series with the total pages visit count.
series = df['TotalVisits']

# Create a list of bin boundaries.
bins = [0, 3, 6, 9, 12]

# Bin the data into the list of bins.
series["bin"] = pd.cut(series, bins)

# Count the number of values in each bin.
counts = series["bin"].value_counts()

# Print the counts.
print(counts)


# In[98]:


series["bin"].value_counts(normalize=True).mul(100).plot(kind = 'bar', figsize = [10,5])
plt.title(" TotalVisits ")


# *The 0 to 3 bin had a higher number of total visits than the other bins

# Page Views Per Visit

# In[99]:


print("NullValues:",df['Page Views Per Visit'].isnull().sum())


# In[100]:


df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].median(), inplace = True)


# In[101]:


print(df['Page Views Per Visit'].value_counts(normalize=True))


# In[102]:


sns.boxplot(df['Page Views Per Visit'])
plt.show()


# In[103]:


df['Page Views Per Visit'].quantile([0.6,0.75,0.95,0.99,1])


# In[104]:


Q11 = df['Page Views Per Visit'].quantile(0.00)
Q14 = df['Page Views Per Visit'].quantile(0.99)
df['Page Views Per Visit'][df['Page Views Per Visit']<= Q11]=Q11
df['Page Views Per Visit'][df['Page Views Per Visit'] >= Q14]=Q14


# In[105]:


sns.boxplot(df['Page Views Per Visit'])
plt.show()


# Inspect Categorical Variable:Last Activity

# In[106]:


100*df['Last Activity'].value_counts(normalize=True,dropna=True)


# In[107]:



print("null values",df['Last Activity'].isnull().sum()/len(df['Last Activity']))


# In[108]:


df.dropna(subset=['Last Activity'],inplace=True)


# In[109]:



print("null values",df['Last Activity'].isnull().sum())
x=df['Last Activity'].value_counts(normalize = True).mul(100)
print(x)


# In[110]:


df['Last Activity'] = df['Last Activity'].replace(list(x[x < 5].index), 'Others')


# In[111]:


df['Last Activity'].value_counts(normalize = True).mul(100) .round(2).plot(kind = 'bar', figsize = [10,5])
plt.xticks(rotation = 45)
plt.xlabel('Customer last event')
plt.ylabel('count of lead')


# In[112]:




plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Last Activity', 'Converted')
plt.xlabel('Customer last event')
plt.ylabel('count of lead')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Last Activity', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)
plt.xlabel('Customer last action')
plt.tight_layout()


# Analysis:8.	Customer with last activity as SMS Sent should be targeted more  since their  conversion rate is 61% and count of those customers are also more.

# Scrutinize each column

# In[113]:


# Inspect: Lead Origin


# In[114]:


print(df['Lead Origin'].value_counts())
print("null values :",df['Lead Origin'].isna().sum())


# In[115]:


df["Lead Origin"]=np.where(df["Lead Origin"].isin(['Landing Page Submission','API']),df["Lead Origin"],"Other Add Form")


# In[116]:


print(100*df['Lead Origin'].value_counts(normalize=True))


# In[117]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Lead Origin', 'Converted')
plt.xticks(rotation = 45)
plt.xlabel('Channels')
plt.ylabel('count of lead')
# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Lead Origin', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)
plt.xlabel('Channels')

plt.tight_layout()


# Analysis : Most Landing Page Submission customer have high numbers are assumed to be potential lead but their conversion rate is 36% .'other Add form' customers have more conversion rate even thoug their count is less and this group should be targetted more

# Inspect : Do Not Email

# In[ ]:


print(df['Do Not Email'].value_counts())
print("Null Values",df['Do Not Email'].isnull().sum())


# In[ ]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Do Not Email', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Do Not Email', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# *.Analysis:Customers who have selected "Do Not Email" as Yes are less likely to be converted

# Inspect : Do Not Call

# In[118]:


print(df['Do Not Call'].value_counts())
print("Null Values",df['Do Not Call'].isnull().sum())
# Dropping this column since the value of yes is too low (i.e., data is skewed)


# In[119]:


df.drop('Do Not Call',axis=1,inplace=True)


# Inspect Target Variable: Converted

# In[120]:


print(df['Converted'].value_counts() )
print("Null Values",df['Converted'].isnull().sum())


# In[121]:


labels=['non_Converted','Converted']
ax=df.Converted.value_counts(normalize=True).plot(kind="pie",ylabel='',labels=labels,legend=True,labeldistance=None,title='Non converted vs converted',autopct='%1.1f%%', radius=1, shadow=False,fontsize=9)
ax.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
plt.show()


# In[122]:


df.columns


# 14.inspect :I agree to pay the amount through cheque','Magazine','Newspaper Article','X Education Forums','Search','Newspaper','Through Recommendations','Update me on Supply Chain Content', 'Get updates on DM Content','Receive More Updates About Our Courses','Through Recommendations','Digital Advertisement'

# In[123]:


# Since these columns contains only one value ,data is skewed so dropping from further analysis 'Do Not Call','Magazine','Newspaper Article','X Education Forums','Search','Newspaper','Through Recommendations','Update me on Supply Chain Content', 'Get updates on DM Content','Receive More Updates About Our Courses','Through Recommendations','Digital Advertisement'
unwanted_col=['Do Not Email','I agree to pay the amount through cheque','Magazine','Newspaper Article','X Education Forums','Search','Newspaper','Through Recommendations','Update me on Supply Chain Content', 'Get updates on DM Content','Receive More Updates About Our Courses','Through Recommendations','Digital Advertisement','Country_new','City_new']
df.drop(unwanted_col, axis = 1, inplace = True)


# In[124]:


df.columns


# 15:Inspect :Lead Profile

# In[125]:


# df['Lead Profile'].isnull().sum()
# df['Lead Profile'].fillna('Potential Lead',inplace=True)


# In[126]:


# #replacing 'Select' with NaN
# df['Lead Profile'].replace({'Select': 'Potential Lead'} ,inplace=True)
# df['Lead Profile'].value_counts()
# # Considering all Leads as Potential Leads


# In[127]:


# plt.figure(figsize = [15,13])
# plt.subplot(211)
# hue_count('Lead Profile', 'Converted')
# plt.xticks(rotation = 45)

# # Rate of conversion
# plt.subplot(212)
# conversion_rate_plot('Lead Profile', 'Converted', p = 'Blues_d')
# plt.xticks(rotation = 45)

# plt.tight_layout()


# *.Analysis:Only 39.3% customers are onverted who are assumed to be potential Lead customers.Dual Specialization Student and Lateral Student will be our next targets after potential Lead customers

# In[128]:


# Checking other columns


# Inspect Numeric :Total Time Spent on Website

# In[129]:


print(df['Total Time Spent on Website'].value_counts())
print("NullValues:",df['Total Time Spent on Website'].isnull().sum())
# No null values found


# In[130]:


# Checking Outliers
sns.boxplot(df['Total Time Spent on Website'])
plt.show()
# No outliers found


# In[131]:


df['A free copy of Mastering The Interview'].value_counts()


# In[132]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('A free copy of Mastering The Interview', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('A free copy of Mastering The Interview', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# Last Notable Activity

# In[133]:


100*df['Last Notable Activity'].value_counts(normalize=True)


# In[134]:


x=100*df['Last Notable Activity'].value_counts(normalize=True)
df['Last Notable Activity']=df['Last Notable Activity'].replace(list(x[x<4].index),"Others")


# In[135]:


df['Last Notable Activity'].value_counts()


# In[136]:


plt.figure(figsize = [15,13])
plt.subplot(211)
hue_count('Last Notable Activity', 'Converted')
plt.xticks(rotation = 45)

# Rate of conversion
plt.subplot(212)
conversion_rate_plot('Last Notable Activity', 'Converted', p = 'Blues_d')
plt.xticks(rotation = 45)

plt.tight_layout()


# Analysis:Last Notable Activity with SMS_SENT are likely to be targetted

# In[137]:



sns.pairplot(df,diag_kind='kde',hue='Converted')
plt.show()


# Data Imbalance:

# In[138]:


df_nonhotleads=df.loc[df["Converted"]==0]
df_hotleads=df.loc[df["Converted"]==1]


# In[139]:


sns.lineplot(data=df_nonhotleads, x='Page Views Per Visit', y='TotalVisits')
plt.ylabel(" TotalVisits")
plt.xlabel("Page Views Per Visit")

plt.xticks(rotation=40)
plt.show()


# In[140]:


sns.lineplot(data=df_hotleads, x='Page Views Per Visit', y='TotalVisits')
plt.ylabel(" TotalVisits")
plt.xlabel("Page Views Per Visit")

plt.xticks(rotation=40)
plt.show()


# In[146]:


import seaborn as sns
# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=[20,5])
# Plot the histograms
axes[0].hist(df['Page Views Per Visit'], bins=30, label='Page Views/Visit',color='#FFFF00')
axes[1].hist(df['TotalVisits'], bins=30, label='TotalVisits',color='#0000FF')
axes[2].hist(df['Total Time Spent on Website'], bins=50, label='Timespent',color='#00FFFF')

# Set the titles of the subplots
axes[0].set_title('Page Views/Visit')
axes[1].set_title('TotalVisits')
axes[2].set_title('Timespent')

# Show the figure
plt.show()


# <!-- Step 2: Visualising/ Data understanding  -->

# In[149]:


#checking corelation
plt.figure(figsize = (20, 12))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show()


# Observations based on above Heatmap
# 
# 1. Positive corelation exist between Page views per visit and target variable "Total Visits".
# 
# 2.No corelation between "Page views per visit" and Target variable "converted" since value is almost almost equalto zero.
# 
# 
# 

# # Data Preparation-
# 1.Create dummy variable for all categorical variable.
# 2.Perform Train test Split (used 70:30)
# 3.Perform Scaling (Used MinMax scaler)

# # Dummy Variable

# In[150]:


# Get the value counts of all the columns

categorical_var = df.loc[:, df.dtypes == 'object']
categorical_var.columns


# In[151]:


# Create dummy variables using the 'get_dummies' command
dummy = pd.get_dummies(df[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',
       'What is your current occupation',
       'What matters most to you in choosing a course', 'Tags',
       'A free copy of Mastering The Interview', 'Last Notable Activity']], drop_first=True)

# Add the results to the master dataframe
df = pd.concat([df, dummy], axis=1)


# In[153]:


df.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',
       'What is your current occupation',
       'What matters most to you in choosing a course', 'Tags',
       'A free copy of Mastering The Interview', 'Last Notable Activity'], axis=1,inplace=True)


# In[154]:


df.describe()


# In[155]:



# Get the value counts of all the columns

for column in df:
    print(df[column].astype('category').value_counts())

    print('***********************************************************')


# In[ ]:


#checking corelation
plt.figure(figsize = (50, 40))
sns.heatmap(df.corr(), annot = True, cmap="OrRd")
plt.show()


# # Test-Train Split

# In[156]:


# Import the required library

from sklearn.model_selection import train_test_split


# In[157]:


# Put all the feature variables in X

x = df.drop(['Converted'], 1)
x.head()


# In[158]:


# Converted is the target variable
y=df['Converted']
y.head()


# In[159]:


# split the date set to 70% and 30%
np.random.seed(0)
x_train, x_test,y_train,y_test  = train_test_split(x,y, train_size = 0.7,test_size = 0.3,random_state = 100)


# In[160]:


print('x_train',x_train.shape)
print('x_test',x_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)


# # Rescaling using min max scaling method

# In[161]:


# Import MinMax scaler

from sklearn.preprocessing import MinMaxScaler


# In[162]:


x_train.describe()


# In[163]:


scaler = MinMaxScaler()


# In[164]:


# Total Time Spent ,Page Views Per Visiton Website,TotalVisit data has high values compared to other columns .so we need to rescale this to fit this in the same range for further analysis
# Apply scaler() to all the columns except the encoding dummy variables
num_vars = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

x_train[num_vars] = scaler.fit_transform(x_train[num_vars])


# In[165]:


x_train.describe()


# In[166]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (50, 40))
sns.heatmap(x_train.corr(), annot = True)
plt.title("Data correlation for training set")
plt.show()


# *.Analysis :Based on above heat map Multicollinearity is present in the data frame

# # Checking the conversion rate

# In[167]:


convert = sum(df['Converted'])/len(df['Converted'])*100
print("conversion rate is ",convert)


# In[168]:


x_train.shape


# # Data Modelling

# In[169]:


# Import RFE
from sklearn.feature_selection import RFE

# RFE with 15 features
lr = LogisticRegression()
rfe = RFE(estimator = lr,n_features_to_select = 15)
rfe = rfe.fit(x_train, y_train)


# In[170]:


rfe.support_


# In[171]:


# features have been selected by RFE

list(zip(x_train.columns, rfe.support_, rfe.ranking_))


# In[172]:


#Columns where RFE support is True
col_RFE_True = x_train.columns[rfe.support_]
col_RFE_True


# In[173]:


#Columns where RFE not supported
col_RFE_False = x_train.columns[~rfe.support_]
col_RFE_False


# In[174]:


# Creating X_test dataframe with RFE selected variables
x_train_rfe = x_train[col_RFE_True]


# In[175]:


x_train_rfe1 = sm.add_constant(x_train_rfe)
x_train_rfe1.head()


# In[176]:


model1 = sm.GLM(y_train, x_train_rfe1, family = sm.families.Binomial())
res = model1.fit()
res.summary()


# In[177]:


# Getting the predicted values on train set
y_train_pred = res.predict(x_train_rfe1)
y_train_pred[:10].values.reshape(-1)


# In[178]:


# creating a dataframe with the actual convert flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'convert_prob': y_train_pred})

y_train_pred_final.head()


# In[179]:


# Creating new column 'predicted' with 1 if Convert_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.convert_prob.map(lambda x:1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[180]:


from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[181]:


#Confusion Metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(confusion)


# In[182]:


# Accuracy
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[183]:


x_train_rfe1.columns


# In[184]:


x_train_rfe1.drop('const',axis=1,inplace=True)


# In[185]:


# Create a dataframe thdt will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_rfe1.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe1.values, i) for i in range(x_train_rfe1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[186]:



x_train_rfe1.columns


# In[187]:


# remodel by dropping Lead Origin_Other Add Form because of high vif and p
x_train_rfe2 = x_train_rfe1.drop(["Lead Origin_Other Add Form"], axis = 1)


# In[188]:


x_train_rfe2.columns


# In[189]:


#Build a model
x_train_rfe3 = sm.add_constant(x_train_rfe2)

model2 = sm.GLM(y_train,x_train_rfe3, family = sm.families.Binomial())
res2 = model2.fit()
res2.summary()


# In[190]:


# Getting the predicted values on train set
y_train_pred = res2.predict(x_train_rfe3)
y_train_pred[:10].values.reshape(-1)


# In[191]:


# creating a dataframe with the actual convert flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'convert_prob': y_train_pred})

y_train_pred_final.head()


# In[192]:


# Creating new column 'predicted' with 1 if Convert_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.convert_prob.map(lambda x:1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[193]:


#Confusion Metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted)
print("Model2 Confusion matrix",confusion)
# Accuracy
print("Model2 Accuracy",100*metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[194]:


x_train_rfe3.columns


# In[195]:


x_train_rfe3.drop('const',axis=1,inplace=True)


# In[196]:


# Create a dataframe thdt will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_rfe3.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe3.values, i) for i in range(x_train_rfe3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[197]:


# Drop Page Views Per Visit due to high VIF


# In[198]:


x_train_rfe3.columns


# In[199]:


x_train_rfe3.drop('Page Views Per Visit',axis=1,inplace=True)


# In[200]:


x_train_rfe3.columns


# In[201]:


#Build a model
x_train_rfe4 = sm.add_constant(x_train_rfe3)

model3 = sm.GLM(y_train,x_train_rfe4, family = sm.families.Binomial())
res2 = model3.fit()
res2.summary()


# In[202]:


# Getting the predicted values on train set
y_train_pred = res2.predict(x_train_rfe4)
y_train_pred[:10].values.reshape(-1)


# In[203]:


# creating a dataframe with the actual convert flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'convert_prob': y_train_pred})

y_train_pred_final.head()


# In[204]:


# Creating new column 'predicted' with 1 if Convert_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.convert_prob.map(lambda x:1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[205]:


#Confusion Metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted)
print(" model3 confusion matrix",confusion)
# Accuracy
print("Accuracy model3",100*metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[206]:


x_train_rfe4.columns


# In[207]:


x_train_rfe4.drop('const',axis=1,inplace=True)


# In[208]:


# Create a dataframe thdt will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_rfe4.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe4.values, i) for i in range(x_train_rfe4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[209]:


# Drop Tags_Will revert after reading the email due to high VIF
x_train_rfe4.columns


# In[210]:


x_train_rfe4.drop('Tags_Will revert after reading the email',axis=1,inplace=True)


# In[211]:


#Build a model
x_train_rfe5 = sm.add_constant(x_train_rfe4)

model4 = sm.GLM(y_train,x_train_rfe5, family = sm.families.Binomial())
res3 = model4.fit()
res3.summary()


# In[212]:


# Getting the predicted values on train set
y_train_pred = res3.predict(x_train_rfe5)
y_train_pred[:10].values.reshape(-1)


# In[213]:


# creating a dataframe with the actual convert flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'convert_prob': y_train_pred})

y_train_pred_final.head()


# In[214]:


# Creating new column 'predicted' with 1 if Convert_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.convert_prob.map(lambda x:1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[215]:


#Confusion Metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted)
print("model 4 confusion matrix ",confusion)


# In[216]:


# Accuracy
print("model 4 Accuracy ",100*metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[217]:


x_train_rfe5.columns


# In[218]:


x_train_rfe5.drop('const',axis=1,inplace=True)


# In[219]:


# Create a dataframe thdt will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = x_train_rfe5.columns
vif['VIF'] = [variance_inflation_factor(x_train_rfe5.values, i) for i in range(x_train_rfe5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[220]:


TP = confusion[1,1] # true positive
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[221]:


# Let's see the sensitivity,specificity of our logistic regression model
# false postive rate - predicting convert when customer does not have converted
print("Final model evaluations")
print('-----------------------------------')
print("Sensitivity",100*TP / float(TP+FN))
print("specificity",100*TN / float(TN+FP))
print("false postive rate",100* FP/ float(TN+FP))
print("positive predictive value ",100*TP / float(TP+FP))
print("Negative predictive value ",100*TN / float(TN+ FN))


# # Plotting the ROC Curve (Optimal curve )

# In[222]:


# ROC CURVE
from sklearn.metrics import roc_curve, roc_auc_score


# In[223]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[224]:


fpr,tpr,threshold=metrics.roc_curve(y_train_pred_final.Converted,y_train_pred_final.convert_prob, drop_intermediate=False)


# In[225]:


draw_roc(y_train_pred_final.Converted,y_train_pred_final.convert_prob)


# # Finding optimal cutoff point

# In[226]:


# Let's create columns with different probability cutoffs
numbers = [float(x)/10 for x in range (10)]
for i in numbers:
    y_train_pred_final[i] = y_train_pred_final.convert_prob.map(lambda x:1 if x>i else 0)
y_train_pred_final.head()


# In[227]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame(columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[228]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


#  Analysis:From the curve above, 0.4 is the optimum point to take it as a cutoff probability.

# In[229]:


y_train_pred_final['final_predicted'] = y_train_pred_final.convert_prob.map( lambda x: 1 if x > 0.4 else 0)

y_train_pred_final.head()


# In[230]:


# Now let us calculate the lead score by round off convert probality

y_train_pred_final['lead_score'] = y_train_pred_final.convert_prob.map(lambda x: round(x*100))
y_train_pred_final.head(20)


# In[231]:


# checking if 80% cases are correctly predicted based on the converted column.

# get the total of final predicted conversion / non conversion counts from the actual converted rates

checking_df = y_train_pred_final.loc[y_train_pred_final['Converted']==1,['Converted','final_predicted']]
checking_df['final_predicted'].value_counts()


# In[232]:


# check the precentage of final_predicted conversions

print("final_predicted conversions",100*2090/float(2090+361))


# In[233]:


# final_predicted conversions is 85% hence this is a good model


# In[234]:


# Let's check the overall accuracy.
print("accuracy",100*metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))


# In[235]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
print("confusion2",confusion2)


# In[236]:


TP = confusion2[1,1] # true positive
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity,specificity of our logistic regression model,false postive rate - predicting convert when customer does not have converted
print("Sensitivity       ",100*TP / float(TP+FN))
print("specificity       ",100*TN / float(TN+FP))
print("false postive rate",100*FP/ float(TN+FP))
print("positive predictive value ",100*TP / float(TP+FP))
print("Negative predictive value ",100*TN / float(TN+ FN))


# # Precision and Recall

# In[239]:


print("accuracy_score",metrics.accuracy_score(y_train_pred_final['Converted'], y_train_pred_final.predicted))
confusion1=metrics.confusion_matrix(y_train_pred_final['Converted'], y_train_pred_final.predicted)
print("confusion_matrix\n",confusion1)


# In[240]:


# Let's evaluate the other metrics as well

TP = confusion1[1,1] # true positive
TN = confusion1[0,0] # true negatives
FP = confusion1[0,1] # false positives
FN = confusion1[1,0] # false negatives


# In[241]:


print("sensitivity",TP/(TP+FN))
print("specificity",TN/(TN+FP))


# In[242]:


from sklearn.metrics import precision_score, recall_score
print("precision score",precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
print("recall score",recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[243]:


# Precision and Recall Tradoff


# In[244]:


from sklearn.metrics import precision_recall_curve
p,r,thresholds = precision_recall_curve(y_train_pred_final.Converted,y_train_pred_final.convert_prob)


# In[245]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# # Making Prediction on Test Set

# In[246]:


num_vars = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']

x_test[num_vars] = scaler.fit_transform(x_test[num_vars])


# In[247]:


x_test.head()


# In[248]:


x_test.describe()


# In[249]:


# list down and check variables of final model
var_final = list(res3.params.index)
var_final.remove('const')
print('Final Selected Variables:', var_final)

# Print the coefficents of final varible
print('\033[1m{:10s}\033[0m'.format('\nCoefficent for the variables are:'))
print(round(res3.params,3))


# In[250]:


# select final variables from X_test
X_test_sm = x_test[var_final]
X_test_sm.head()


# In[251]:


X_test_sm.columns


# In[252]:


X_train_sm = sm.add_constant(X_test_sm)


# In[253]:


X_train_sm.head()


# In[254]:


x_train_rfe5.head()


# In[255]:


x_train_rfe5.head()


# In[256]:


# predict test dataset
y_test_pred = res3.predict(X_train_sm)


# In[257]:




y_test_pred[:10].values.reshape(-1)


# In[258]:


# Converting y_pred to a dataframe which is an array
y_pred_df = pd.DataFrame(y_test_pred)


# In[259]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[260]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)
y_pred_final.head()


# In[261]:


# Renaming the column
y_pred_final= y_pred_final.rename(columns={ 0 : 'convert_prob'})


# In[262]:


y_pred_final.head()


# In[263]:


y_pred_final['final_predicted'] = y_pred_final.convert_prob.map(lambda x: 1 if x > 0.40 else 0)


# In[264]:


y_pred_final.head()


# In[265]:


# Now let us calculate the lead score

y_pred_final['lead_score'] = y_pred_final.convert_prob.map(lambda x: round(x*100))
y_pred_final.head()


# In[266]:


# checking if cases are correctly predicted based on the converted column.

# get the total of final predicted conversion or non conversion counts from the actual converted rates

checking_test_df = y_pred_final.loc[y_pred_final['Converted']==1,['Converted','final_predicted']]
checking_test_df['final_predicted'].value_counts()


# In[267]:


# check the precentage of final_predicted conversions on test data
print("Test data final_predicted conversions on test data",100*843/float(843+158))


# In[268]:


print("accuracy",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted))
confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
print("confusion of test data",confusion2)
TP = confusion2[1,1] # true positive
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity,specificity,false postive rate - predicting convert when customer does not have converted of our logistic regression model

print("Test data Sensitivity       ",100*TP / float(TP+FN))
print("Test data specificity       ",100*TN / float(TN+FP))
print("Test data false postive rate",100*FP/ float(TN+FP))
print("Test data positive predictive value ",100*TP / float(TP+FP))
print("Test data Negative predictive value ",100*TN / float(TN+ FN))


# In[269]:


train_result = dict([
    ("final_predicted conversions", 85.27),
    ("false positive rate", 14.51),
    ("Accuracy", 85.40),
    ("Sensitivity", 85.27),
    ("specificity", 85.48),
])


# In[270]:


test_result = dict([
    ("final_predicted conversions", 84.21),
    ("false positive rate", 14.16),
    ("Accuracy", 85.24),
    ("Sensitivity", 84.21),
    ("specificity", 85.83),
])


# In[271]:


train_result = pd.Series(train_result)


# In[272]:


test_result = pd.Series(test_result)


# In[273]:


Final_Solution=pd.concat([train_result,test_result],axis=1,keys=['Train_result','Test_result'])


# In[275]:



print("ANALYSIS OF THE Lead Scoring Model .Based on the below prediction we have the folloowing stats.")
print("-------------------------------------------------------------------------------------------")
print(Final_Solution)


# # Conclusion:
# 1.	Logistic regression model is used to predict the probability of conversion of customer.
# 2.	Optimum cut-off chosen to be 0.40:
#     •	A lead score of 0.40 or higher indicates a hot lead.
#     •	Leads with a score of less than 0.40 are cold leads. These leads may still convert, but they're less likely to do           so. You can place them in a nurturing program to help them move closer to conversion
# 3.	Our final model is built based on 12 features
# 
# 4.	The company should focus on the following to make the best use of its time and resources:
#     •	Source Welingak Website/ Reference/Olark Chat
#     •	last action as SMS Conversation,
#     •	Working Professionals since their  conversion rate is  92%
#     •	More Total Visits and Total Time Spent on Website.
# 5.	Sales team should ignore Leads:
#     •	Who are Specialized in Travel and Tourism
#     •	Customers who have recently received an SMS are more likely to convert, so we should target them more
#     •	With motto other than "Better career" prospects.
#     •	Should not consider leads with current status “Ringing.
# 6.	Company should focus on Welingak Website. They are the most likely to convert, so they are the best use of companies time and resources.
# 7.	Google is a major source of leads, but we should focus our efforts on leads from Welingak Website or Reference because they are more likely to convert.
# 8.	Customers who have opted out of email are less likely to become paying customers.
# 9.	Landing Page Submission customers are a large group, but they have a lower conversion rate than 'Other Add form' customers. 'Other Add form' customers are a smaller group, but they have a higher conversion rate, so we should focus our efforts on targeting them
# 10.	The Management sector is a high-converting sector, so we should focus our efforts on targeting leads from this sector.
# 11.	Customers who have indicated that they will revert after reading the email are more likely to convert than other customers in this group.
# 
# 

# Equation:
# Converted=-1.884+(1.177*TotalVisits')+(1.177 *'Total Time Spent on Website')+(4.69*'Lead Source_Olark Chat')+(1.566* 'Lead Source_Reference')+(4.111*'Lead Source_Welingak Website')+(5.821*'Last Activity_Olark Chat Conversation')+(-0.869*'Last Activity_Others')+(-0.722*'Specialization_Travel and Tourism')+(2.096*'What is your current occupation_Working Professional')+(-1.748*'What matters most to you in choosing a course_Other')+(-4.570*'Tags_Ringing')+(1.864*'Last Notable Activity_SMS Sent')
# 
# 

# In[ ]:




