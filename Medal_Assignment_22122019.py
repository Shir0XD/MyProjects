#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import statistics as st
import os

os.chdir('C:\\Users\\Shishir\\Desktop\\Desktop\\contact session 2\\code and datasets\\EDA day 1')
medal=pd.read_csv('medal.csv',sep=',',header=0, encoding="latin")

os.chdir('C:\\Users\\Shishir\\Desktop\\Desktop\\')
mapping=pd.read_csv('mapping2.csv',sep=',',header=0, encoding="latin")

medal = pd.merge(medal,mapping,how='left',on='NOC')
medal.columns


# In[19]:


#Some Stats
unique_women = len(medal[medal.Gender=='Women'].Athlete.unique())
print('Count of unique Women= ',unique_women)
unique_men = len(medal[medal.Gender=='Men'].Athlete.unique())
print('Count of unique men= ',unique_men)
women_medals = medal[medal.Gender=='Women'].Medal.count()
print('Medals won by Women= ',women_medals)

#Women gold Medals
women_medals = medal[medal.Gender=='Women']
women_gold_medals = women_medals[women_medals.Medal=='Gold']
women_gold_medals = women_gold_medals.Medal.count()
print('Gold Medals won by Women= ',women_gold_medals)

men_medals = medal[medal.Gender=='Men'].Medal.count()                                                        
print('Medals won by Men= ',men_medals)

#Men Gold Medals
men_medals = medal[medal.Gender=='Men']
men_gold_medals = men_medals[men_medals.Medal=='Gold']
men_gold_medals = men_gold_medals.Medal.count()
print('Gold Medals won by Men= ',men_gold_medals)

Total_Events = medal.Event.nunique()
Total_NOC = medal.NOC.nunique()
Total_Sport = medal.Sport.nunique()
Total_Discipline = medal.Discipline.nunique()

print('Count of unique countries= ',Total_NOC)
print('Count of unique sports= ',Total_Sport)
print('Count of unique disciplines= ',Total_Discipline)
print('Count of unique events= ',Total_Events)


# In[86]:


#Event Stats
print(medal[medal.Medal.fillna('None')!='None'].Medal.value_counts())
# How many total medals.
medal[medal.Medal.fillna('None')!='None'].shape[0]


# In[2]:


#Top 10 counries by Gold
Gold_by_NOC = medal[medal['Medal']=='Gold'].groupby(['Country'])['Medal'].agg(['count']).sort_values("count",ascending = False)
top10NOC = medal[medal['Medal']=='Gold'].groupby(['Country'])['Medal'].agg(['count']).sort_values("count",ascending = False).head(10)
top10NOC.plot(kind = 'bar')
print(top10NOC)


# In[3]:


#Bottom10 countries
bottom10NOC = medal[medal['Medal']=='Gold'].groupby(['Country'])['Medal'].agg(['count']).sort_values("count",ascending = True).head(10)
print(bottom10NOC)
bottom10NOC.plot(kind = 'bar')


# In[4]:


#Which country has won the most gold medals?
medal[medal['Medal']=='Gold'].groupby(['Country'])['Medal'].agg(['count']).sort_values("count",ascending = False).head(10)


# In[5]:


#Which athlete has won the most gold medals?
medal[medal['Medal']=='Gold'].groupby(['Country','Athlete'])['Medal'].agg(['count']).sort_values("count",ascending = False).head(10)


# In[6]:


#Cities hosting most editions
medal.groupby(['City'])['Edition'].nunique().sort_values(ascending = False).head(10)


# In[5]:


#Outlier analysis on counts
Country_medal_count = medal.groupby(['Country']).Medal.agg('count')
Country_medal_count = Country_medal_count.reset_index(name='count').sort_values(['count'], ascending=False)
Country_medal_count.columns

OutlierMedals = Country_medal_count['count']
q1=OutlierMedals.quantile(0.25) # finding the lower quartile value 
q3=OutlierMedals.quantile(0.75) # finding the upper quartile value 
iqr = q3-q1  # Calculating the inter quartile range
print("q1=",q1)
print("q3=",q3)
print("iqr=",iqr)
upperfence = q3+(1.5*iqr) 
lowerfence = q1-(1.5*iqr)
print("upperfence=",upperfence)
print("lowerfence=",lowerfence)
a = [num for num in OutlierMedals if num > upperfence]
b = [num2 for num2 in OutlierMedals if num2 < lowerfence]
outlrs=a+b
print("Outliers = ",outlrs)

Outliers = Country_medal_count[Country_medal_count['count']>upperfence]
Outliers

#why are these countries not winning as much medals?
Outliers.plot(x='Country', kind = 'bar')
Country_medal_count.reset_index(inplace=True)
Country_medal_count.drop(columns='index',axis=1).head(10)


# In[ ]:





# In[18]:


#Which athlete has most Gold Medals(Men)
filter1 = medal["Medal"]=="Gold"
filter2 = medal["Gender"]=="Men"
Gold_by_Men = medal.where(filter1 & filter2)
Gold_by_Men.dropna(inplace=True)
Gold_by_Men
Gold_by_Men.groupby(['Athlete','Sport','Discipline'])['Medal'].agg(['count']).sort_values("count",ascending = False).head(10)


# In[35]:


#Diving deep into swimming
filter1 = medal["Medal"]=="Gold"
filter2 = medal["Gender"]=="Men"
filter3 = medal["Discipline"] == "Swimming"
Men_Swimmers = medal.where(filter1 & filter2 & filter3)
Men_Swimmers.dropna(inplace=True)
Men_Swimmers = Men_Swimmers.groupby(['Edition','Athlete'])['Medal'].agg(['count']).sort_values("count",ascending = False)
Men_Swimmers = Men_Swimmers.groupby(['Edition','Athlete']).agg({'count':np.max}).sort_values(by = 'count',ascending = False).groupby(level=0).head(1)
Men_Swimmers = Men_Swimmers.sort_values(by='Edition',ascending = True)
Men_Swimmers = Men_Swimmers.reset_index(inplace = False)
print(Men_Swimmers)
Men_Swimmers.plot(x='Edition', kind = 'bar')

#Which Athlete has max number of golds is swimming in each edition?


# In[36]:


#Which athlete has most Gold Medals(Women)
filter1 = medal["Medal"]=="Gold"
filter2 = medal["Gender"]=="Women"
Gold_by_Women = medal.where(filter1 & filter2)
Gold_by_Women.dropna(inplace=True)
Gold_by_Women = Gold_by_Women.groupby(['Athlete','Sport','Discipline'])['Medal'].agg(['count']).sort_values("count",ascending = False).head(10)
Gold_by_Women


# In[18]:


print(medal[medal.Gender=='Women'].Edition.min())
print(medal[medal.Gender=='Men'].Edition.min())


# In[182]:


Men_by_Year = medal[medal.Gender== 'Men'].groupby('Edition').agg('count').Athlete
women_by_Year = medal[medal.Gender=='Women'].groupby('Edition').agg('count').Athlete
print(Men_by_Year)
print(women_by_Year)
(sns.scatterplot(data= Men_by_Year),
 sns.scatterplot(data=women_by_Year))


# In[189]:


Gold_by_Year = medal[medal.Medal== 'Gold'].groupby('Edition').agg('count').Medal
Silver_by_Year = medal[medal.Medal=='Silver'].groupby('Edition').agg('count').Medal
Bronze_by_Year = medal[medal.Medal=='Bronze'].groupby('Edition').agg('count').Medal

(sns.scatterplot(data= Gold_by_Year,color='orange'),
 sns.scatterplot(data= Silver_by_Year,color = 'green'),
sns.scatterplot(data= Bronze_by_Year, color = 'brown'))


# In[20]:


#Max Medals for each edition
Counties_Medal = medal.groupby(['Edition','Country'])['Medal'].agg(['count']).sort_values("count",ascending = False)
Counties_Medal.reset_index(inplace = True)
Counties_Medal = Counties_Medal.groupby(['Edition','Country']).agg({'count':np.max}).sort_values(by = 'count',ascending = False).groupby(level=0).head(1)
Counties_Medal = Counties_Medal.reset_index()
Counties_Medal = Counties_Medal.sort_values(by = 'Edition',ascending = True)
Counties_Medal.reset_index(drop = True, inplace=True)
Counties_Medal


# In[84]:


f, ax = plt.subplots(figsize=(8, 6)) 
fig = sns.boxplot(x='NOC', y="count", data=Counties_Medal)
fig.axis(ymin=0, ymax=500)


# In[38]:


medal_analytics = medal.copy()
medal_analytics.Medal[medal_analytics.Medal == 'Gold'] = 100
medal_analytics.Medal[medal_analytics.Medal == 'Silver'] = 0.1
medal_analytics.Medal[medal_analytics.Medal == 'Bronze'] = 0.2
medal_analytics.Gender[medal_analytics.Gender == 'Men'] = 0.1
medal_analytics.Gender[medal_analytics.Gender == 'Women'] = 100

NOC_with_NoGold = medal_analytics.groupby(['Country']).Medal.agg(['sum'])
NOC_with_NoGold[NOC_with_NoGold['sum']<100]
#Why are these countries not able to win any gold medals?


# In[39]:


NOC_with_NoWomen = medal_analytics.groupby(['Country']).Gender.agg(['sum'])
NOC_with_NoWomen[NOC_with_NoWomen['sum']<100]
#why these coutries have no women  participation?


# In[8]:


event_group = medal.groupby(['Edition','Event'])['Event'].agg(['count'])
event_group.reset_index(inplace=True)
event_group = event_group.iloc[:,[0,1]]
event_group = event_group.groupby(['Edition','Event'])['Event'].agg(['count'])
event_group.reset_index(inplace=True)
event_group = event_group.groupby(['Edition'])['Event'].agg(['count'])
event_group.plot(kind = 'line')
#events were dropped around 1920, which were those events?


# In[10]:


#Scorecard
medal_count = medal.groupby(['Edition','Country','Medal']).Medal.agg('count')

medal_count = medal_count.reset_index(name='count').sort_values(['count'], ascending=False)
#medal_count.head(40)


table = pd.pivot_table(medal_count, values='count', index=['Edition', 'Country'],columns=['Medal'], aggfunc=np.sum)
table


# In[51]:


#Country wise Scorecard with Function

countries = medal_count['Country'].unique()
countries = countries.tolist()
        
def get_NOC_stats(Country):
    for x in countries:
        if Country == x:
            return pd.pivot_table(medal_count[medal_count.Country==Country], values='count', index=['Country', 'Edition'],columns=['Medal'], aggfunc=np.sum, margins=True)
                    
       
    print('Country not found, please enter one of the below countries:' )
    print(sorted(countries))
        
get_NOC_stats('United States of America')


# In[66]:


#Edition wise scorecard with function

editions = medal_count['Edition'].unique()
editions = editions.tolist()

def get_Edition_stats(Ed):
    
    for x in editions:
        if Ed == x:
            pt = pd.pivot_table(medal_count[medal_count.Edition==Ed], values='count', index=['Edition', 'Country'],columns=['Medal'], aggfunc=np.sum, margins=True,margins_name='All')
            return  pt
              
    print('Edition not found, please enter one of the below Editions:' )
    print(sorted(editions))
    
get_Edition_stats(1948)


# In[56]:


medal.columns =[column.replace(" ", "_") for column in medal.columns]
medal.query('Gender == "Men" and Event_gender == "W"')


# In[57]:


print(sorted(countries))


# In[62]:


sport = medal['Sport'].unique()
sport = sport.tolist()
print(sport)
def Sport(x):
    return medal[medal['Sport']== x].groupby(['Sport','Discipline','Event','Edition'])[["Edition"]].count()

Sport('Gymnastics')


# In[60]:




