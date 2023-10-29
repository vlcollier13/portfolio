#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:23:04 2022

@author: vanessa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:36:57 2022

@author: vanessa
"""


#%% LIBRARIES
#%%% PANDAS

import pandas as pd
from pandas import Series, DataFrame
from pandas import DataFrame as df
#%%% MATPLOTLIB
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib import pyplot
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#%%% Numpy
import numpy as np
from numpy import nan as NA

#%%% Seaborn
import seaborn as sns
sns.set(color_codes=True)

#%%% Plotly
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly.figure_factory as ff
pio.renderers.default='browser'
# pio.renderers.default='svg'

#%%% Scipy
import scipy.stats as st
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import normaltest
from scipy.stats import norm

#%%% STATSMODELS
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsap
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_white
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

#%%% OTHER
import statistics
import dtale
import os
from os.path import expanduser as ospath
from prettytable import PrettyTable
from stargazer.stargazer import Stargazer

#%%% SKLEARN

import sklearn
import sklearn.utils._cython_blas
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


#%%% LINEAR MODELS
import linearmodels
from linearmodels.panel import PooledOLS
from linearmodels.panel import compare
from linearmodels.panel import PanelOLS
from linearmodels.panel import RandomEffects
from linearmodels.datasets import wage_panel

#%%% SET THEMES
sns.set_theme(font_scale = 1.5)


#%% DATA IMPORT

path = os.chdir('/Users/Vanessa/Library/Mobile Documents/com~apple~CloudDocs/GRAD SCHOOL - iCLOUD/SPRING 2022/PYTHON/PYTHON_FINAL_PROJECT/AUTOCRACY/VDEM')

# Remove scientific notation globally

pd.options.display.float_format = '{:,.4f}'.format

# Import VDEM data
vdem_original = pd.read_csv("VDEM.csv")
# To display the top 5 rows 
vdem_original.head(10) 

# Datatypes

vdem_original.dtypes

# Select only needed variables

vdem = vdem_original[["country_name","year","v2x_regime","v2caautmob","v2smgovdom","v2smgovab","v2smpardom","v2smparab","v2smfordom","v2smforads", "e_regionpol", "e_gdppc", "v2smpolsoc"]]
vdem.head(10) 


# Limit to 1950, post WWII period to present

type(vdem.year)

vdem = vdem[vdem['year'].between(1950,2021)]

# Check for nulls
print(vdem.isnull().sum())


# Dropping the missing values.

vdem = vdem.dropna()    
vdem.count()

# After dropping the values

print(vdem.isnull().sum())  



#%% RECODING
# RECODING SECTION ###########################################################
# PURPOSE: For ease of interpretation and comparision with target variable (mobilization for autocracy),
# Recode disinformation variables so that scale is reversed making target "Disinformation_median"  


def reverseScoring(df, i):
    '''Reverse scores on given columns
     df = your data frame,
     cols = the columns you want reversed in list form'''
    df[i] = max(df[i]) - df[i]
    return df


#  The Columns to be reversed
cols_rev = ["v2smgovdom"],["v2smgovab"],["v2smpardom"],["v2smparab"],["v2smfordom"],["v2smforads"]

cols_rev= list(cols_rev)


# Find max values

for [i]in cols_rev:
    print ('Maximum is:', max(vdem[i]))

# as loop
result = [] 
for [i]in cols_rev:
    result.append((max(vdem[i])))

# apply to vdem and create new df with reversed scores
for [i] in cols_rev:
    revFrame = reverseScoring(vdem, i)
    
    
# Compare with vdem visually - Belgium should be at zero disinformation, Turkmenistan should have most disinformation.

# adopt changes

vdem=revFrame


# Combine disinformation variables into one aggregate 

vdem['disinformation_agg'] = vdem["v2smgovdom"]+vdem["v2smgovab"] + vdem["v2smpardom"] + vdem["v2smparab"] + vdem["v2smfordom"] + vdem["v2smforads"]

vdem['disinformation_median'] = vdem['disinformation_agg']/6


#%%% Disinformation Groups
quant_range = vdem['disinformation_median'].quantile([0.10,0.25,0.5,0.75, 0.90])

quant_range.reset_index(drop=True)
quant_10 = quant_range.iloc[0:1]
quant_10 = float(quant_10)
quant_25 = quant_range.iloc[1:2]
quant_25 = float(quant_25)
quant_50 = quant_range.iloc[2:3]
quant_50 = float(quant_50)

quant_75 = quant_range.iloc[3:4]
quant_75 = float(quant_75)

quant_90 = quant_range.iloc[4:5]
quant_90 = float(quant_90)

# Create function for disinformation groups


conds = [vdem.disinformation_median.between(0,quant_10), vdem.disinformation_median.between(quant_10,quant_25),
         vdem.disinformation_median.between(quant_25,quant_50), vdem.disinformation_median.between(quant_50,quant_75),
         vdem.disinformation_median.between(quant_75,quant_90)]


choices = [0,1,2,3,4]


vdem['disinfo_level'] = np.select(conds,choices, 5)

# COUNTS
disinfo_country_count = vdem['disinfo_level'].value_counts()

##############################################################################
#%% DATA DESCRIPTION
# Find the dimensions of the dataset

vdem.shape

# Feature Names
list(vdem.columns)

# Data types of features
vdem.dtypes

# Summarize 
summary = vdem.describe()
summary=round(summary, 3)
print(summary)

# Count Unique Values
vdem.nunique()


# Add column for new labels
vdem['Regime_Type'] = vdem['v2x_regime'].replace({0: 'Closed Autocracy',
                                                              1: 'Electoral Autocracy',
                                                                  2: 'Electoral Democracy',
                                                                      3: 'Liberal Democracy'})

# select most important variables 

vdem_simple = vdem

vdem_simple = vdem[['country_name','disinformation_median','disinfo_level','v2caautmob', 'v2smpolsoc', 'e_gdppc', 'year', 'Regime_Type','e_regionpol']] 

# Convert to categorical variables

for col in ['e_regionpol','Regime_Type', 'country_name']:
    vdem_simple[col] = vdem_simple[col].astype('category')
vdem_simple.dtypes

from pandas.api.types import CategoricalDtype
# Create ordered categorical for 'disinfo_level'
vdem_simple['disinfo_level'] = vdem_simple['disinfo_level'].astype(CategoricalDtype(ordered=True))

# Classify Features
target = "v2caautmob"

numeric_features = list(vdem_simple.select_dtypes("float64").columns)
numeric_features.remove(target)
categorical_features = list(vdem_simple.select_dtypes("category").columns)

print(f'numeric_features:\n{numeric_features}\n\ncategorical_features:\n{categorical_features}\n\ntarget:\n{target}')

vdem_simple.dtypes

#%%% Indexing for panel regression

# Convert year to integer
vdem_simple.year = vdem.year.astype(np.int64)

# Multi index, entity - time
vdem_simple = vdem_simple.set_index(['country_name', 'year'], drop = False)


#%% DATA DISTRIBUTION
#%%% HISTOGRAMS
matplotlib.use('TkAgg')

#%%%% Multiple histograms

for i in vdem_simple:
    fig = px.histogram(vdem, x = i, title = "Distribution 2000 - 2021")
    fig.show()


#%%% Bar Plot 

data_query_2019_highest = vdem_simple[(vdem_simple['year'] == 2019) & (vdem_simple['disinfo_level'] >= 5)]

#count observations grouped by team and division
counts = vdem.groupby(['year', 'Regime_Type','disinfo_level']).size().reset_index(name='obs')

# Query by year

data_query_2000 = counts[(counts['year'] == 2000)]
data_query_2019 = counts[(counts['year'] == 2019)]

# Bar Chart - all years

fig = px.bar(counts, x="year", y="obs", color="disinfo_level", barmode="group",
             facet_col="Regime_Type", title = "Disinformation by Regime Type by Year")
fig.show()

# Bar Chart 2000
fig = px.bar(data_query_2000, x="year", y="obs", color="disinfo_level", barmode="group",
             facet_col="Regime_Type", title = "Disinformation by Regime in 2000",
             labels = {'x':"Fruits<br><sup>Fruit sales in the month of January</sup>", 
              'y':'count'})

fig.show()


# Bar Chart 2021

fig = px.bar(data_query_2019, x="year", y="obs", color="disinfo_level", barmode="group",
             facet_col="Regime_Type", title = "Disinformation by Regime Type in 2019")
fig.show()

# Countries with highest disinfo in 2019
fig = px.bar(data_query_2019_highest, x="country_name", y="disinformation_median", 
             color="Regime_Type", barmode="group",title = "Countries with Highest Disinformation 2019")
fig.show()
    



#%%% Features

# Detect high correlation features 

plt.figure(figsize = (12, 8))
sns.heatmap(vdem.corr(), annot = True, cmap = 'viridis_r');
plt.title("Correlation Heatmap")

high_corr_features = ["v2x_regime", "v2caautmob","v2smpolsoc", "disinformation_median"]

#%%% Scatter matrix for feature overview

pd.plotting.scatter_matrix(vdem[high_corr_features], figsize = (12, 10))
  

#%%% Log Transformation
#%%%% v2caautmob (target/dv)
f, ax = plt.subplots(1,2)

ax[0].hist(vdem.v2caautmob)
ax[0].set_title('v2caautmob')
ax[1].hist(np.log(vdem.v2caautmob))
ax[1].set_title("Log v2caautmob")

#%%%% disinformation_median
f, ax = plt.subplots(1,2)

ax[0].hist(vdem.disinformation_median)
ax[0].set_title('disinformation_median')
ax[1].hist(np.log(vdem.disinformation_median))
ax[1].set_title("Log disinformation_median")

#%%%% Create new logged variables
vdem['LOG_disinformation_median'] = np.log(vdem.disinformation_median)

vdem['LOG_v2caautmob'] = np.log(vdem.v2caautmob)


#%%%% Create feats and eng_feats

feats = ['disinformation_median', 'v2caautmob', 'v2smpolsoc', "e_gdppc"]

eng_feats = ['LOG_disinformation_median', 'LOG_v2caautmob', 'v2smpolsoc']

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))
sns.heatmap(vdem[feats].corr(), annot = True, cmap = 'viridis_r', ax = ax1).set(title = 'original data')
sns.heatmap(np.log(vdem[eng_feats]).corr(), annot = True, cmap = "viridis_r", ax = ax2).set(title = "transformed data")



#%% DESCRIPTIVE STATISTICS
#%% DESCRIBE
vdem_describe2 = vdem_simple.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
vdem_describe2.to_excel("vdem_describe2.xlsx")

#%%% Frequencies
frequency = vdem_simple.describe()

# Find frequency of each regime across all years
crosstab_vdem = pd.crosstab(index=vdem_simple['Regime_Type'], columns='count')

# Select path for export
frequency.to_excel(r'path\Frequency.xlsx', index = True)

#%%% Pair plots

#1
fig_scatmatrix = px.scatter_matrix(vdem, 
                        dimensions=["v2caautmob", "disinformation_median", "v2smpolsoc", "e_gdppc"],
                        color_discrete_sequence=['dodgerblue'], title = "Scatterplot Matrix of Variables")


fig_scatmatrix.add_annotation(text="Source: VDEM Institute",
                  xref="paper", yref="paper",
                  x=0.0, y=1.0, showarrow=False)


fig_scatmatrix.show()

#%%% Correlation Matrix

vdem[['disinformation_median','v2caautmob','v2smpolsoc',"e_gdppc" ]].corr()

caption = "Source: Varieties of Democracy (VDEM)"

#plot the correlation matrix 
plt.title("Correlation Heatmap", fontsize =20)
plt.figtext(0.5, -0.8, caption, wrap=True, horizontalalignment='center', fontsize=12)

sns.heatmap(vdem[['disinformation_median','v2caautmob','v2smpolsoc','e_gdppc']].corr(), 
annot=True, cmap = 'Reds')

plt.show()

#%%% BOX PLOTS


df = vdem_simple

fig = px.box(df, x="year", y="disinformation_median", color="Regime_Type", 
             title = "Disinformation Median by Regime Type",
 hover_data=["country_name"])
             
fig.show()


#%%% SCATTERPLOTS
#%%%% Scatterplot Disinformation median vs. mobilization for autocracy

df = vdem_simple
fig = px.scatter(df, x="disinformation_median", y="v2caautmob", trendline="ols",color = "year",
                title="Growth of Disinformation and Mobilization for Autocracy")

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))   


fig.add_annotation(text="Source: VDEM Institute",
                  xref="paper", yref="paper",
                  x=0.0, y=0.0, showarrow=False)

fig.show() 

#%%%% Scatterplot Faceted by Regime Type with trendline
fig_scatter = px.scatter(vdem_simple, x="disinformation_median", y="v2caautmob", 
                         title = "Scatterplot of Regime Type with Trendline", 
                         facet_col="Regime_Type", color="year", trendline="ols",
                         category_orders={"Regime_Type": ["Closed Autocracy", "Electoral Autocracy", "Electoral Democracy", "Liberal Democracy"]})
fig_scatter.update_layout(
    font=dict(size = 20))
fig_scatter.add_annotation(text="Source: VDEM Institute",
                  xref="paper", yref="paper",
                  x=0.0, y=0.0, showarrow=False)



fig_scatter.show()

trendlines = px.get_trendline_results(fig_scatter)

print(trendlines)


#%% TIME SERIES 
#%%% Time Series- Regime Types


# Group by year and count regime types per year
group_by_year = vdem.groupby(["year", "Regime_Type"], as_index=False)["country_name"].count()
group_by_year['Count of Countries']= group_by_year['country_name']
print(group_by_year.dtypes)

fig = px.line(group_by_year, x="year", y="Count of Countries", color='Regime_Type', title='Regime Types by Year',

)
fig.update_layout(
    font=dict(size = 18),
    annotations = [dict(
        x=0.0,
        y=-0.1,    
        xref='paper',
        yref='paper',
        text='Source: VDEM Institute',
        showarrow = False
    )]
)

fig.show()

#%%%  United States Disinfo Timeline

data_query_US = vdem_simple[(vdem_simple['country_name'] == 'United States of America')]


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=data_query_US['year'], y=data_query_US['disinformation_median'], name="Disinformation Median"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=data_query_US['year'], y=data_query_US['v2caautmob'], name="Mobilization for Autocracy"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Disinformation and Mobilization for Autocracy in the United States"
)

# Set x-axis title
fig.update_xaxes(title_text="Year")

# Set y-axes titles
fig.update_yaxes(title_text="<b>primary</b> Disinformation Median", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> Mobilization for Autocracy", secondary_y=True)

fig.show()
fig.update_layout(
    annotations = [dict(
        x=0.0,
        y=-0.1,    
        xref='paper',
        yref='paper',
        text='Source: VDEM Institute',
        showarrow = False
    )]
)


fig.show()


####################

df1 = data_query_US

df2 = data_query_US

sns.lineplot(data=df1, x='year', y='disinformation_median', color="red")
sns.lineplot(data=df2, x='year', y='v2caautmob')

#%%% Time Series - Autocracy

group_by_year_auto = vdem.groupby(["year", "Regime_Type"], as_index=False)["v2caautmob"].median()


fig = px.line(group_by_year_auto, x="year", y="v2caautmob", color='Regime_Type', title='Mobilization for Autocracy by Regime Type per year',

)
fig.update_layout(
    annotations = [dict(
        x=0.5,
        y=-0.1,    
        xref='paper',
        yref='paper',
        text='Source: VDEM Institute',
        showarrow = False
    )]
)


fig.show()


#%% MAPS
import plotly.graph_objects as go
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/1962_2006_walmart_store_openings.csv')
df.head()

#%%% ANIMATED CHOROPLETH

# Import country code 
import plotly.graph_objects as go
import pandas as pd
from urllib.request import urlopen
import json

countrycode_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
	

# Add country code to dataframe
px.choropleth(vdem_simple, 
              locations = 'country_name',
              color="disinformation_median", 
              animation_frame="year",
              color_continuous_scale="Inferno",
              locationmode='country names',
              scope="world",
              range_color=(0, 5),
              title='Progression of Disinformation',
              height=800
             )

px.choropleth(vdem_simple, 
              locations = 'country_name',
              color="v2caautmob", 
              animation_frame="year",
              color_continuous_scale="Inferno",
              locationmode='country names',
              scope="world",
              range_color=(-3,3),
              title='Progression of Mobilization for Autocracy',
              height=800
             )

#%% PANEL REGRESSION 


from stargazer.stargazer import Stargazer, LineLocation
import statsmodels.graphics.tsaplots as tsap
from statsmodels.stats.diagnostic import het_white
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col
import statsmodels.formula.api as smf
from linearmodels.panel import compare


#%%% Pooled OLS
from linearmodels import PooledOLS

y = vdem_simple['v2caautmob']
x = vdem_simple[['disinformation_median', 'v2smpolsoc', 'e_gdppc']]
mod1 = PooledOLS(y, x)
res1 = mod1.fit(cov_type="clustered", cluster_entity=True)

print(res1)

# Pooled OLS 2
from linearmodels import PooledOLS
from linearmodels.iv.results import compare

y = vdem_simple['v2caautmob']
x = vdem_simple[['disinformation_median', 'e_gdppc']]
mod2 = PooledOLS(y, x)
res2 = mod2.fit(cov_type="clustered", cluster_entity=True)

print(res2)

# Pooled OLS 3
from linearmodels import PooledOLS
from linearmodels.iv.results import compare

y = vdem_simple['v2caautmob']
x = vdem_simple[['disinformation_median']]
mod3 = PooledOLS(y, x)
res3 = mod3.fit(cov_type="clustered", cluster_entity=True)

print(res3)

#%%% Entity Effects

mod_entity = PanelOLS(vdem_simple.v2caautmob, vdem_simple[['disinformation_median', 'e_gdppc']], entity_effects=True)
res_entity = mod_entity.fit(cov_type='clustered', cluster_entity=True)

print (res_entity)

#%%% Time Effects
mod_time = PanelOLS(vdem_simple.v2caautmob, vdem_simple[['disinformation_median','e_gdppc']], time_effects=True)
res_time = mod_time.fit(cov_type='clustered', cluster_entity=True)

print(res_time)

#%%% Entity and Time Effects
mod_te = PanelOLS(vdem_simple.v2caautmob, vdem_simple[['disinformation_median','e_gdppc']], entity_effects = True, time_effects=True)
res_te = mod_te.fit(cov_type='clustered', cluster_entity=True)

print(res_te)


#%%% COMPARE MODELS


from linearmodels.panel import compare
from linearmodels.panel import results
from linearmodels.panel import model
from linearmodels.panel.model import PanelEffectsResults
from linearmodels.panel.model import PanelResults
from statsmodels.iolib import summary


comparison = compare({"Model 1": res1,"Model 2": res2, "Model 3": res3 , "Model 4": res_entity, "Model 5": res_time, "Model 6": res_te}, precision= 'pvalues',stars =True)
table = {
    '(1)': res1,
    '(2)': res2,
    '(3)': res3, 
    '(4)': res_entity,
    '(5)': res_time,
    '(6)': res_te
}


comparison = compare(table,precision= 'pvalues',stars =True)
summary = comparison.summary
html12 = summary.as_html()

f = open("results12.html", "w")
f.write(html12)
f.close()


#%% REGRESSION ASSUMPTIONS
#%%% Linearity

from statsmodels import tsa
from statsmodels import tools
import statsmodels.stats.diagnostic
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from linearmodels.compat import statsmodels
from statsmodels.iolib import summary


# Repeat Model 5 to plot residuals
y = vdem_simple['v2caautmob']
x = vdem_simple[['disinformation_median', 'e_gdppc']]

mod5 = PanelOLS(vdem_simple.v2caautmob, vdem_simple[['disinformation_median','e_gdppc']], time_effects=True)
mod5_fit = mod5.fit(cov_type="clustered", cluster_entity=True)


# Plot Residuals

# Obtain residuals
print('Parameters: ', mod5_fit.params)
print('R2: ', mod5_fit.rsquared)
print('Residuals', mod5_fit.resids)


# Produce regression plots

# create a DataFrame of predicted values and residuals
vdem_simple["predicted"] = mod5_fit.predict(x)
vdem_simple["residuals"] = mod5_fit.resids

#define figure size
fig = plt.figure(figsize=(12,8))


# Residuals plot
sns.scatterplot(data=vdem_simple, x="predicted", y="residuals").set(title = "Model 5: Predicted Values vs. Residuals")
plt.axhline(y=0)

#%%%% QQPLOT check for normal distribution
# QQ Plots


qqplot_vdem3=qqplot(mod5_fit.resids,line='s').gca().lines 


#%%% Autocorrelation
import matplotlib.pyplot as plt

# Durbin-Watson 
from statsmodels.stats.stattools import durbin_watson

# Durbin-Watson test
durbin_watson(mod5_fit.resids)

#%%% No Multicollinearity among Predictors

# NA when v2smpolsoc and egdppc removed from model

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

a = vdem_simple['disinformation_median']
b = vdem_simple['e_gdppc']
c = vdem_simple['v2smpolsoc']


ck = np.column_stack([a, b, c])
cc = np.corrcoef(ck, rowvar=False)
VIF = np.linalg.inv(cc)
VIF.diagonal()

#%%% HOMOSKEDASTICITY


# Perform PooledOLS
from linearmodels import PooledOLS
import statsmodels.api as sm

# Convert year
vdem_simple['year'] = pd.Categorical(vdem_simple['year'], ordered=True)

exog = sm.tools.tools.add_constant(vdem_simple['disinformation_median'])
endog = vdem_simple['v2caautmob']
mod = PooledOLS(endog, exog)
pooledOLS_res = mod.fit(cov_type='clustered', cluster_entity=True)

# Store values for checking homoskedasticity graphically
fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
residuals_pooled_OLS = pooledOLS_res.resids


# 3A. Homoskedasticity
import matplotlib.pyplot as plt

 # 3A.1 Residuals-Plot for growing Variance Detection
fig, ax = plt.subplots()
ax.scatter(fittedvals_pooled_OLS, residuals_pooled_OLS, color = 'blue')
ax.axhline(0, color = 'r', ls = '--')
ax.set_xlabel('Predicted Values', fontsize = 15)
ax.set_ylabel('Residuals', fontsize = 15)
ax.set_title('Homoskedasticity Test', fontsize = 30)
plt.show()



#%%% ENDOGENEITY # NOT YET WORKING CORRECTLY

import linearmodels.iv.model as lm
from linearmodels.iv.model import IVResults

#y = vdem_simple['v2caautmob']
#x = vdem_simple[['disinformation_median', 'e_gdppc']]

#mod5 = PanelOLS(vdem_simple.v2caautmob, vdem_simple[['disinformation_median','e_gdppc']], time_effects=True)
#mod5_fit = mod5.fit(cov_type="clustered", cluster_entity=True)

#constant = sm.add_constant(data = vdem_simple, prepend = False )

#%%% GITHUB
#git init

  




