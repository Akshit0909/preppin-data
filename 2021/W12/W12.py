#!/usr/bin/env python
# coding: utf-8

# # DATA PREP

# ----

# In[1]:


# Preppin Data 2021 W12
import pandas as pd
import numpy as np
import re


# In[81]:


#Input Data
df=pd.read_csv("Input/Tourism Input.csv")
df.head()


# In[3]:


# Pivot all of the month fields into a single column. The columns to stay intact are mentioned in id_vars
df=pd.melt(df,id_vars=df.iloc[:,0:4],var_name='month')
df.head()


# In[4]:


#Rename Columns
df.rename(columns={'Series-Measure':'measure','Hierarchy-Breakdown':'breakdown','Unit-Detail':'unit'},inplace=True)
df.head()


# In[5]:


#Test Example: Remove extraneous characters from numbers
pd.Series(["78%",'na','12.2%']).map(lambda x: re.sub("na|%","",x))


# In[6]:


#Removing na and % from numbers
df['value']=df['value'].map(lambda x: re.sub("%|na","",x))


# In[7]:


#Filter out the blanks
df=df[df['value']!=""]


# In[8]:


df.info()


# In[9]:


#ensure that each field has the correct data type
df['unit']=df['unit'].astype('category')
df['month']=pd.to_datetime(df['month'],format='%b-%y')
df['value']=df['value'].astype('float')
df.info()


# In[10]:


df.head()


# In[11]:


'''
Our goal now is to remove all totals and subtotals from our dataset so that only the lowest 
level of granularity remains. Currently we have Total > Continents > Countries, but we don't 
have data for all countries in a continent, so it's not as simple as just filtering out the 
totals and subtotals. Plus in our Continents level of detail, we also have The Middle East and 
UN passport holders as categories. If you feel confident in your prep skills, this (plus the output) 
should be enough information to go on, but otherwise read on for a breakdown of the steps we need to take:
'''

#Filter our dataset so our Values are referring to Number of Tourists
df=df[df['unit']=='Tourists']
df.head()


# In[12]:


#Filter out Total tourist arrivals
df=df[~df.measure.str.contains('Total')]


# In[13]:


df.shape


# In[14]:


df['measure'].value_counts()


# In[15]:


df[['breakdown','measure']].value_counts()


# In[16]:


'''
Strategy: We have data at 2 level of granularities mixed up.
1. country level: we have data by continent(breakdown) X country(measure)
2. continent level: we have continents in the (measure) columns

We make separate df for both of these using "breakdown" column.
Now in country level: we don't have data for all countries in a continent
Eg: In country data: it's like Europe->Germany 100 tourists, Europe->France 200 tourists
    But in continent data we have Europe->1000 visitors
    So we have 700 tourists for which we have country as "Unknown"
    
Strategy step 2: 
Aggregate country data to continent level and join it with actual continent data to find out
the number of tourists for which we have do not have country information(700 in our example)
          
After doing this, join this back to country data.
'''


# In[17]:


#Split our workflow into 2 streams: Continents and Countries
df_continent=df[df['breakdown']=='Real Sector / Tourism / Tourist arrivals']
df_country=df[df['breakdown']!='Real Sector / Tourism / Tourist arrivals']


# In[18]:


df_country.head()


# In[19]:


df_continent.head()


# In[20]:


#Split out the Continent and Country names from the relevant fields
continents = ['Europe','Asia','Africa','Americas','Oceania','Middle East','UN passport holders and others']
countries =['Germany','Italy','Russia','United Kingdom','China','India','France','Australia','United States']

continents_list = '|'.join(continents)
countries_list = '|'.join(countries)


# In[21]:


#Test Example: How search function in regex works to find out string from a list of values separated by pipes.
#We can use start() and end() function to get values of tuple positions
re.search("Akshit|Neel|Hi|Bye|Asia", "Tourist arrivals from Asia")


# In[22]:


def find_string_in_list(string_search:str, list_in_pipes:str):
    position=re.search(list_in_pipes,string_search) #This finds string in the list which has pipes and returns a tuple of position
    if position:
        return string_search[position.start():position.end()]
    else:
        return "NA"


# In[23]:


df_continent['continent']=df_continent.measure.apply(lambda x:find_string_in_list(string_search=x,list_in_pipes=continents_list))
df_continent


# In[24]:


df_country['continent']=df_country.breakdown.apply(lambda x:find_string_in_list(string_search=x,list_in_pipes=continents_list))
df_country['country']=df_country.measure.apply(lambda x:find_string_in_list(string_search=x,list_in_pipes=countries_list))
df_country


# In[25]:


df_continent.drop(['measure','breakdown','unit'],axis=1,inplace=True)
df_country.drop(['measure','breakdown','unit'],axis=1,inplace=True)


# In[26]:


#Aggregate our Country stream to the Continent level 
df_country_agg=df_country.groupby(['month','continent'],as_index=False)['value'].agg("sum")
df_country_agg


# In[27]:


df_country_agg.rename(columns={"value":"country_agg_value"},inplace=True)


# In[28]:


#Join the two streams together and work out how many tourists arrivals there are that we don't know the country of
data_join=pd.merge(df_continent,df_country_agg,how="left",on=['month','continent'])


# In[29]:


data_join.fillna({'country_agg_value':0,'value':0},inplace=True)
data_join


# In[30]:


data_join['difference']=data_join['value']-data_join['country_agg_value']
data_join


# In[31]:


#Add in a Country field with the value "Unknown"
data_join['country']='Unknown'
data_join


# In[32]:


data_join=data_join[['id','month','difference','continent','country']].rename(columns={'difference':'value'})
data_join


# In[33]:


df_country


# In[34]:


#Union this back to here we had our Country breakdow
result_df=pd.concat([df_country,data_join])
result_df


# In[35]:


#Data cleansing: remove leading and trailing spaces
result_df[['continent','country']]=result_df[['continent','country']].apply(lambda x:x.str.strip(), axis=1)


# In[36]:


result_df=result_df[['month','continent','country','value']].rename(columns={'continent':'breakdown','value':'number of tourists'})
result_df.head()


# In[82]:


#Output Data
result_df.to_csv("Output/AM_Tourism_Output.csv",index=False)


# In[38]:


#Method 2: Didn't go with this but this is how we can take values from one column and put it in another based on a condition
#in Pandas

'''
df['continent updated']=df.apply(lambda x: x.country if x.continent == " Tourist arrivals" else x.continent, axis=1)
df['country']=df.apply(lambda x: "Unknown" if x.continent==" Tourist arrivals" else x.country, axis=1)
df.head()
'''


# # DATA VIZ

# ----

# In[39]:


import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns
import matplotlib.style as style
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize']=8,4
import warnings
warnings.filterwarnings('ignore')


# In[40]:


df_viz=result_df.copy()


# In[41]:


#Converting month to mmm-yy (use strftime to convert datetime to string)
df_viz['month']=df_viz['month'].dt.strftime('%b-%y')
df_viz


# In[42]:


#Making year column
df_viz['year']="20"+df_viz['month'].apply(lambda x: x.split("-")[1])
df_viz


# In[43]:


df_viz.info()


# In[44]:


df_viz.breakdown.unique()


# In[45]:


#Checking where the volumne of tourists come from
df_viz.groupby(['breakdown'])['number of tourists'].agg("sum").plot.barh(x="breakdown",y="number of tourists")


# In[46]:


#How number of tourists have changed through years : Line chart
df_viz.groupby(['year'])['number of tourists'].agg("sum").plot.line(x="breakdown",y="number of tourists")


# In[47]:


#Distribution of number of tourists by years: Violin Plot: Tells us the mean, ditribution as well as density
fig,ax=plt.subplots()
fig.set_size_inches(15,7)
sns.violinplot(data=df_viz,
            x="year",y="number of tourists")


# In[48]:


df_viz


# In[49]:


# Attempt to make a stream graph
x=range(2010,2021)
asia_plot=df_viz[df_viz['breakdown']=='Asia'].groupby('year',as_index=False)['number of tourists'].agg('sum').iloc[:,1]
europe_plot=df_viz[df_viz['breakdown']=='Europe'].groupby('year',as_index=False)['number of tourists'].agg('sum').iloc[:,1]
plt.stackplot(x,asia_plot,europe_plot,
             labels=['Asia','Europe'],baseline="sym")


# In[50]:


#Stream Graph Attempt 2
from scipy.interpolate import make_interp_spline
tnew = np.linspace(2010, 2020, num=100, endpoint=True)
f = make_interp_spline(x, asia_plot, k=2)
y1Smooth = f(tnew)

f = make_interp_spline(x, europe_plot, k=2)
y2Smooth = f(tnew)

plt.stackplot(tnew, y1Smooth, y2Smooth, labels=['Asia','Europe'], baseline='sym')


# In[51]:


#Pivoting the dataframe, tranforming continents to columns for data viz ease (Not used though)
df_viz_pvt=df_viz.pivot_table(index="year",columns='breakdown',values='number of tourists',aggfunc='sum')
df_viz_pvt


# In[52]:


#Heat Map: Distribution by year and continent
sns.heatmap(df_viz_pvt, cmap='viridis')


# #### AGGREGATED DATA BY CONTINENT & YEAR

# In[53]:


#Aggregate data by continent and year for data viz and YOY calculations for each continent
df_viz_contyear=df_viz.groupby(['breakdown','year'],as_index=False)['number of tourists'].agg('sum')
df_viz_contyear


# In[54]:


#Making a custom dict to sort the data by the below order
#WHY? Because Facet Grid takes the order of dataframe to plot graphs and we need the order from maximum YOY% to minimum YOY%
#This order has been taken from YOY% dataframe
custom_dict = {'Oceania':0,
 'Asia':1,
 'UN passport holder and others':2,
 'Africa':3,
 'Americas':4,
 'Europe':5,
 'Middle East':6}


# In[55]:


#Custom Sorting
df_viz_contyear.sort_values(by=['breakdown','year'], key=lambda x: x.map(custom_dict),inplace=True)
df_viz_contyear


# In[56]:


#Removing UN Passport Countries as that won't be the focus for Viz
df_viz_contyear=df_viz_contyear[~df_viz_contyear['breakdown'].str.contains('UN passport')]


# #### YOY % CHANGE

# In[57]:


df_viz_yoy = df_viz_contyear.copy()


# In[58]:


df_viz_yoy['YOY'] = df_viz_yoy['number of tourists'].shift(1)
df_viz_yoy.head()


# In[59]:


df_viz_yoy['YOY%'] = ( (df_viz_yoy['number of tourists'] - df_viz_yoy['YOY'])/df_viz_yoy['YOY'] )*100
df_viz_yoy.head()


# In[60]:


df_viz_yoy['YOY%']=df_viz_yoy['YOY%'].round(2)
df_viz_yoy.head()


# In[61]:


df_viz_yoy.year=df_viz_yoy.year.astype(str)


# In[62]:


df_viz_yoy=df_viz_yoy[df_viz_yoy['year']=="2020"]
df_viz_yoy.head()


# In[63]:


df_viz_yoy.reset_index(drop=True,inplace=True)
df_viz_yoy.head()


# In[64]:


df_viz_yoy.sort_values(by=['YOY%'],inplace=True)
df_viz_yoy.head()


# In[65]:


YOYP_list=df_viz_yoy['YOY%'].tolist()
YOYP_list


# In[66]:


continent_list=df_viz_yoy['breakdown'].str.upper().tolist()
continent_list


# In[67]:


#List of Styles available in matplotlib
style.available


# In[68]:


#List of Font Families available
import matplotlib
[f.name for f in matplotlib.font_manager.fontManager.afmlist]


# In[69]:


#Global Settings: Applies to every chart
plt.rcParams["font.family"] = "Times New Roman"


# In[70]:


#reset properties of seaborn
sns.set()


# In[80]:


#Convert year to string as it will be plotted as X-axis
df_viz_contyear['year']=df_viz_contyear['year'].astype(str)

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot') #Enhances the charts

#Choose style and don't show grid lines
sns.set_style("whitegrid",{'axes.grid' : False})

#Make facetgrids: Auto multiple charts by categories

'''
Parameters of FacetGrid: 
#Color coding: Add "_r" at end of palettes to reverse the color coding
col_wrap: How many charts in one row
size: Size of each chart: Size will change the height, while maintaining the aspect ratio (so it will also also get wider if only size is changed.)
sharex=False: Don't keep the x axis same for all charts
aspect: How much proportion width should be w.r.t to height
Eg: if height is 80 and aspect is 1.2 then width of chart will be 80*1.2
Official Ex: Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches.
'''

g=sns.FacetGrid(df_viz_contyear,col="breakdown",hue='breakdown',palette='Purples_r',
                col_wrap=3,size=4,aspect=1.1,sharex=False)

#Line Chart
g.map(plt.plot,"year","number of tourists")

#Area Chart
g.map(plt.fill_between,"year","number of tourists",alpha=0.2)

#Spacing/Padding between charts
g.fig.tight_layout(h_pad=3,w_pad=5)


#Setting x tick labels: At position 0 and 10 show 2010 and 2020
g.set(xticks=[0,10], xticklabels=[2010, 2020])
g.set_xticklabels(size = 10)

#Function to format y axis tick labels: Takes as input value,position
#Note: We don't really do anything with position

def y_fmt(x,y):
    return '{:,.0f}'.format(x/1000) + 'K'

k=0

#Loop to go through each chart
for ax in g.axes:
    #Make Reference line
    ax.axvline(9,ls="--",color='black',linewidth=1)
    
    #Don't show any labels: As we're only showing 2010 and 2020 and it has been set before
    ax.set(xlabel=None)
    
    #We disbaled grids at the start, so only enabling Y grid as we don't need X grid(vertical grids)
    ax.yaxis.grid(True,linewidth=0.4)
    
    #Set custom colors: Not using
    #ax.set_color(colors[k])
    
    #Font size of yticks labels
    ax.set_yticklabels(ax.get_yticks(),fontsize=13)
    
    #Format y tick labels: make them in Ks
    ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt),)
    
    #Custom dynamic title of every chart
    ax.set_title(continent_list[k]+" | "+'{:,.1f}'.format(YOYP_list[k])+"%",size=14,weight="bold")
    
    #Show Tick marks
    #https://matplotlib.org/3.1.0/gallery/ticks_and_spines/major_minor_demo.html
    
    k=k+1
    
#Show 2019 just after reference line ONLY in first chart
#https://stackoverflow.com/questions/13413112/creating-labels-where-line-appears-in-matplotlib-figure
g.axes[0].text(8.3,0.45,'Covid-19 Start',rotation=90,transform=g.axes[0].get_xaxis_transform(),size=13)

#Main Title: Top right most position is 1,1. Changing Y adds padding of titles with charts
g.fig.suptitle("MALDIVES' TOURISM BEARS THE BRUNT OF COVID-19",size=20,y=1.15,weight='bold',color="black",)

#Sub Title Hack: Position of sub title has been done by custom x and y
plt.figtext(0.5, 1.06, '( % change in number of tourists from 2019 to 2020 )', ha='center', va='center',
          size=18,color="gray")

#Saving plot as svg file
# bbox_inches='tight': Without this, the titles won't come in the saved file. dpi is for high resolution
plt.savefig('W12.svg',format='svg', dpi=1200,bbox_inches='tight' )

#plt.show() should come after plt.savefig()
#Explanation: plt.show() clears the whole thing, so anything afterwards will happen on a new empty figure

plt.show()


# In[72]:


#Attempt to make a reference line
plt.plot((2010,2010),(2010,2020),c="gray",ls="--") #(x1,x2) (y1,y2)


# In[73]:


#Exploring content of g.axes(multiple charts in the facetgrid)
g.axes


# In[74]:


#Area Chart: Number of tourists in Asia by Year
plt.fill_between(x,asia_plot)


# In[75]:


#Swarm Plot: Number of tourists by country & year
sns.swarmplot(data=df_viz,y='number of tourists',x='year',hue='breakdown')

