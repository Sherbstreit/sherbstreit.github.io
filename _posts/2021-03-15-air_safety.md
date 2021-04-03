---
title: "Airplane Safety"
date: 2021-03-15
tags: [data wrangling, data science, airplane safety, visualization]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Analyzing airplane accident data to evaluate the safety of flying"
mathjax: "true"
---

<iframe src='https://view.officeapps.live.com/op/embed.aspx?src=[https://github.com/Sherbstreit/sherbstreit.github.io/blob/master/images/640_final.pptx]' width='100%' height='600px' frameborder='0'>



```python
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_excel
import seaborn as sns
```


```python
df = pd.read_excel('airline_countries.xls')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>airline</th>
      <th>avail_seat_km_per_week</th>
      <th>incidents_85_99</th>
      <th>fatal_accidents_85_99</th>
      <th>fatalities_85_99</th>
      <th>incidents_00_14</th>
      <th>fatal_accidents_00_14</th>
      <th>fatalities_00_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ireland</td>
      <td>Aer Lingus</td>
      <td>320906734</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Russia</td>
      <td>Aeroflot*</td>
      <td>1197672318</td>
      <td>76</td>
      <td>14</td>
      <td>128</td>
      <td>6</td>
      <td>1</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Argentina</td>
      <td>Aerolineas Argentinas</td>
      <td>385803648</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mexico</td>
      <td>Aeromexico*</td>
      <td>596871813</td>
      <td>3</td>
      <td>1</td>
      <td>64</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>Air Canada</td>
      <td>1865253802</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pycountry 
# create country codes to use geospatial data in Tableau
def alpha3code(column):
    CODE=[]
    for country in column:
        try:
            code=pycountry.countries.get(name=country).alpha_3
           # .alpha_3 means 3-letter country code 
            CODE.append(code)
        except:
            CODE.append('None')
    return CODE
# create a column for country code 
df['CODE']=alpha3code(df.Country)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>airline</th>
      <th>avail_seat_km_per_week</th>
      <th>incidents_85_99</th>
      <th>fatal_accidents_85_99</th>
      <th>fatalities_85_99</th>
      <th>incidents_00_14</th>
      <th>fatal_accidents_00_14</th>
      <th>fatalities_00_14</th>
      <th>CODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ireland</td>
      <td>Aer Lingus</td>
      <td>320906734</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IRL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Russia</td>
      <td>Aeroflot*</td>
      <td>1197672318</td>
      <td>76</td>
      <td>14</td>
      <td>128</td>
      <td>6</td>
      <td>1</td>
      <td>88</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Argentina</td>
      <td>Aerolineas Argentinas</td>
      <td>385803648</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>ARG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mexico</td>
      <td>Aeromexico*</td>
      <td>596871813</td>
      <td>3</td>
      <td>1</td>
      <td>64</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>MEX</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>Air Canada</td>
      <td>1865253802</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>CAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
import geopandas
# first merge geopandas data with my data
# 'naturalearth_lowres' is geopandas dataset
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# rename the columns so that I can merge with my data
world.columns=['pop_est', 'continent', 'name', 'CODE', 'gdp_md_est', 'geometry']
#merge with my data 
merge=pd.merge(world,df,on='CODE')
merge.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pop_est</th>
      <th>continent</th>
      <th>name</th>
      <th>CODE</th>
      <th>gdp_md_est</th>
      <th>geometry</th>
      <th>Country</th>
      <th>airline</th>
      <th>avail_seat_km_per_week</th>
      <th>incidents_85_99</th>
      <th>fatal_accidents_85_99</th>
      <th>fatalities_85_99</th>
      <th>incidents_00_14</th>
      <th>fatal_accidents_00_14</th>
      <th>fatalities_00_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35623680</td>
      <td>North America</td>
      <td>Canada</td>
      <td>CAN</td>
      <td>1674000.0</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -122.9742...</td>
      <td>Canada</td>
      <td>Air Canada</td>
      <td>1865253802</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>326625791</td>
      <td>North America</td>
      <td>United States of America</td>
      <td>USA</td>
      <td>18560000.0</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -120.0000...</td>
      <td>United States</td>
      <td>Alaska Airlines*</td>
      <td>965346773</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>326625791</td>
      <td>North America</td>
      <td>United States of America</td>
      <td>USA</td>
      <td>18560000.0</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -120.0000...</td>
      <td>United States</td>
      <td>American*</td>
      <td>5228357340</td>
      <td>21</td>
      <td>5</td>
      <td>101</td>
      <td>17</td>
      <td>3</td>
      <td>416</td>
    </tr>
    <tr>
      <th>3</th>
      <td>326625791</td>
      <td>North America</td>
      <td>United States of America</td>
      <td>USA</td>
      <td>18560000.0</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -120.0000...</td>
      <td>United States</td>
      <td>Delta / Northwest*</td>
      <td>6525658894</td>
      <td>24</td>
      <td>12</td>
      <td>407</td>
      <td>24</td>
      <td>2</td>
      <td>51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>326625791</td>
      <td>North America</td>
      <td>United States of America</td>
      <td>USA</td>
      <td>18560000.0</td>
      <td>MULTIPOLYGON (((-122.84000 49.00000, -120.0000...</td>
      <td>United States</td>
      <td>Hawaiian Airlines</td>
      <td>493877795</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# group by continent and save as csv to use in Tableau
grouped_cont = merge.groupby('continent').sum()
grouped_cont.to_csv('by_continent.csv', index=True)
merge.to_csv('by_continent_full.csv', index=True)
```


```python
# number weeks/year = 52.1429
# conversion factor from km to mile = 0.62137
# divided by 1 million
df['mil_miles_year_per_seat'] = (df['avail_seat_km_per_week'] * 52.1429 * 0.62137) / 1000000
```


```python
# took mean number of fatalities over 14 year periods divided 
# by the number of miles flown
df['fatality_per_mil_mile_85_99'] = (df['fatalities_85_99']/14) / df['mil_miles_year_per_seat']
df['fatality_per_mil_mile_00_14'] = (df['fatalities_00_14']/14) / df['mil_miles_year_per_seat']
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>airline</th>
      <th>avail_seat_km_per_week</th>
      <th>incidents_85_99</th>
      <th>fatal_accidents_85_99</th>
      <th>fatalities_85_99</th>
      <th>incidents_00_14</th>
      <th>fatal_accidents_00_14</th>
      <th>fatalities_00_14</th>
      <th>CODE</th>
      <th>mil_miles_year_per_seat</th>
      <th>fatality_per_mil_mile_85_99</th>
      <th>fatality_per_mil_mile_00_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ireland</td>
      <td>Aer Lingus</td>
      <td>320906734</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>IRL</td>
      <td>10397.389020</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Russia</td>
      <td>Aeroflot*</td>
      <td>1197672318</td>
      <td>76</td>
      <td>14</td>
      <td>128</td>
      <td>6</td>
      <td>1</td>
      <td>88</td>
      <td>None</td>
      <td>38804.623552</td>
      <td>0.000236</td>
      <td>0.000162</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Argentina</td>
      <td>Aerolineas Argentinas</td>
      <td>385803648</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>ARG</td>
      <td>12500.051225</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mexico</td>
      <td>Aeromexico*</td>
      <td>596871813</td>
      <td>3</td>
      <td>1</td>
      <td>64</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>MEX</td>
      <td>19338.666899</td>
      <td>0.000236</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>Air Canada</td>
      <td>1865253802</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>CAN</td>
      <td>60434.286180</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
mean_mile_85 = (df['fatality_per_mil_mile_85_99'].mean())
mean_mile_00 = (df['fatality_per_mil_mile_00_14'].mean())
```


```python
lst = ['Between 1985 and 1999', 'Between 2000 and 2014'] 
  
# list of int 
lst2 = [mean_mile_85, mean_mile_00] 
  
# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
fatal_by_mile = pd.DataFrame(list(zip(lst, lst2)), 
               columns =['Timeframe', 'Fatalities per Million Miles']) 
fatal_by_mile
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timeframe</th>
      <th>Fatalities per Million Miles</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Between 1985 and 1999</td>
      <td>0.000286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Between 2000 and 2014</td>
      <td>0.000143</td>
    </tr>
  </tbody>
</table>
</div>




```python
# convert to csv format to use in Tableau
fatal_by_mile.to_csv('fatal_mile.csv', index=False)
```


```python
# filter for U.S. data only
airlines_us_df = df[(df.Country == 'United States')]
airlines_us_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>airline</th>
      <th>avail_seat_km_per_week</th>
      <th>incidents_85_99</th>
      <th>fatal_accidents_85_99</th>
      <th>fatalities_85_99</th>
      <th>incidents_00_14</th>
      <th>fatal_accidents_00_14</th>
      <th>fatalities_00_14</th>
      <th>CODE</th>
      <th>mil_miles_year_per_seat</th>
      <th>fatality_per_mil_mile_85_99</th>
      <th>fatality_per_mil_mile_00_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>United States</td>
      <td>Alaska Airlines*</td>
      <td>965346773</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>88</td>
      <td>USA</td>
      <td>31277.268048</td>
      <td>0.000000</td>
      <td>0.000201</td>
    </tr>
    <tr>
      <th>11</th>
      <td>United States</td>
      <td>American*</td>
      <td>5228357340</td>
      <td>21</td>
      <td>5</td>
      <td>101</td>
      <td>17</td>
      <td>3</td>
      <td>416</td>
      <td>USA</td>
      <td>169398.954393</td>
      <td>0.000043</td>
      <td>0.000175</td>
    </tr>
    <tr>
      <th>19</th>
      <td>United States</td>
      <td>Delta / Northwest*</td>
      <td>6525658894</td>
      <td>24</td>
      <td>12</td>
      <td>407</td>
      <td>24</td>
      <td>2</td>
      <td>51</td>
      <td>USA</td>
      <td>211431.568557</td>
      <td>0.000137</td>
      <td>0.000017</td>
    </tr>
    <tr>
      <th>26</th>
      <td>United States</td>
      <td>Hawaiian Airlines</td>
      <td>493877795</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>USA</td>
      <td>16001.657238</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>United States</td>
      <td>Southwest Airlines</td>
      <td>3276525770</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>USA</td>
      <td>106159.545606</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# average number of fatalities/year between 2000-2014
us_each00 = (airlines_us_df['fatalities_00_14'].sum())/14
```


```python
# link to population info
# https://www.multpl.com/united-states-population/table/by-year
us_pop_07 = 301230000
us_pop_17 = 325100000
# women of childbearing age (15-49)
# https://www.who.int/data/maternal-newborn-child-adolescent-ageing/indicator-explorer-new/mca/women-of-reproductive-age-(15-49-years)-population-(thousands)
us_pop_18_childbearing = 74774530
```


```python
# used 2007 as midpoint data for flight data ranging from 2000-14
odds_us_fly_death = us_each00 / us_pop_07
odds_us_fly_death
```




    1.629035241225262e-07




```python
# lightening data average in US between 2009-2018
# https://www.weather.gov/safety/lightning-odds
odds_lightening = 1/1222000
odds_lightening
```




    8.183306055646482e-07




```python
# https://www.statista.com/statistics/527321/deaths-due-to-choking-in-the-us/
odds_choking = 5216 / us_pop_17
odds_choking
```




    1.6044294063365117e-05




```python
# https://www.cdc.gov/nchs/data/nvsr/nvsr69/nvsr69-13-508.pdf
odds_childbirth = 658 /us_pop_18_childbearing
odds_childbirth
```




    8.79978784219707e-06




```python
# link to death by fire 
# https://www.usfa.fema.gov/data/statistics/#tab-2
odds_fire = 3400 / us_pop_17
odds_fire
```




    1.0458320516764072e-05




```python
lst = ['Airplane Crash', 'Lightening Strike', 'Choking', 'Maternal Death in Childbirth', 'Burned in Fire'] 
  
# list of int 
lst2 = [odds_us_fly_death, odds_lightening, odds_choking, odds_childbirth, odds_fire] 
  
# Calling DataFrame constructor after zipping 
# both lists, with columns specified 
compare_df = pd.DataFrame(list(zip(lst, lst2)), 
               columns =['Death due to', 'Likelihood per person']) 
compare_df.to_csv('death_odds.csv', index=False)
compare_df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Death due to</th>
      <th>Likelihood per person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Airplane Crash</td>
      <td>1.629035e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lightening Strike</td>
      <td>8.183306e-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Choking</td>
      <td>1.604429e-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Maternal Death in Childbirth</td>
      <td>8.799788e-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Burned in Fire</td>
      <td>1.045832e-05</td>
    </tr>
  </tbody>
</table>
</div>




```python
cars_data = pd.read_csv('table_02_17_121019.csv')  
cars_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Table 2-17:  Motor Vehicle Safety Data</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>...</th>
      <th>Unnamed: 26</th>
      <th>Unnamed: 27</th>
      <th>Unnamed: 28</th>
      <th>Unnamed: 29</th>
      <th>Unnamed: 30</th>
      <th>Unnamed: 31</th>
      <th>Unnamed: 32</th>
      <th>Unnamed: 33</th>
      <th>Unnamed: 34</th>
      <th>Unnamed: 35</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1960</td>
      <td>1965</td>
      <td>1970</td>
      <td>1975</td>
      <td>1980</td>
      <td>1985</td>
      <td>1990</td>
      <td>1991</td>
      <td>1992</td>
      <td>...</td>
      <td>2009</td>
      <td>2010</td>
      <td>2011</td>
      <td>2012</td>
      <td>2013</td>
      <td>2014</td>
      <td>2015</td>
      <td>2016</td>
      <td>(R) 2017</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fatalities</td>
      <td>36,399</td>
      <td>47,089</td>
      <td>52,627</td>
      <td>44,525</td>
      <td>51,091</td>
      <td>43,825</td>
      <td>44,599</td>
      <td>41,508</td>
      <td>39,250</td>
      <td>...</td>
      <td>33,883</td>
      <td>32,999</td>
      <td>32,479</td>
      <td>33,782</td>
      <td>32,893</td>
      <td>32,744</td>
      <td>35,484</td>
      <td>37,806</td>
      <td>37,473</td>
      <td>36,560</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Injured persons</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>(R) 3,246,000</td>
      <td>(R) 3,107,000</td>
      <td>(R) 3,079,000</td>
      <td>...</td>
      <td>(R) 2,224,000</td>
      <td>(R) 2,248,000</td>
      <td>(R) 2,227,000</td>
      <td>(R) 2,369,000</td>
      <td>(R) 2,319,000</td>
      <td>(R) 2,343,000</td>
      <td>(R) 2,455,000</td>
      <td>(R) 3,062,000</td>
      <td>2,745,000</td>
      <td>2,710,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Crashes</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>6,471,000</td>
      <td>6,117,000</td>
      <td>6,000,000</td>
      <td>...</td>
      <td>5,505,000</td>
      <td>5,419,000</td>
      <td>5,338,000</td>
      <td>5,615,000</td>
      <td>5,687,000</td>
      <td>6,064,000</td>
      <td>6,296,000</td>
      <td>6,821,000</td>
      <td>6,453,000</td>
      <td>6,734,000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Vehicle-miles (millions)</td>
      <td>718,763</td>
      <td>887,811</td>
      <td>1,109,724</td>
      <td>1,327,664</td>
      <td>1,527,295</td>
      <td>1,774,826</td>
      <td>2,144,362</td>
      <td>2,172,050</td>
      <td>2,247,151</td>
      <td>...</td>
      <td>2,956,764</td>
      <td>2,967,266</td>
      <td>2,950,402</td>
      <td>2,969,433</td>
      <td>2,988,280</td>
      <td>3,025,656</td>
      <td>3,095,373</td>
      <td>3,174,408</td>
      <td>3,212,347</td>
      <td>3,240,327</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
# rename columns
cars_data.columns = cars_data.iloc[0]
cars_data = cars_data[1:5]
cars_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NaN</th>
      <th>1960</th>
      <th>1965</th>
      <th>1970</th>
      <th>1975</th>
      <th>1980</th>
      <th>1985</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>...</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>(R) 2017</th>
      <th>2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Fatalities</td>
      <td>36,399</td>
      <td>47,089</td>
      <td>52,627</td>
      <td>44,525</td>
      <td>51,091</td>
      <td>43,825</td>
      <td>44,599</td>
      <td>41,508</td>
      <td>39,250</td>
      <td>...</td>
      <td>33,883</td>
      <td>32,999</td>
      <td>32,479</td>
      <td>33,782</td>
      <td>32,893</td>
      <td>32,744</td>
      <td>35,484</td>
      <td>37,806</td>
      <td>37,473</td>
      <td>36,560</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Injured persons</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>(R) 3,246,000</td>
      <td>(R) 3,107,000</td>
      <td>(R) 3,079,000</td>
      <td>...</td>
      <td>(R) 2,224,000</td>
      <td>(R) 2,248,000</td>
      <td>(R) 2,227,000</td>
      <td>(R) 2,369,000</td>
      <td>(R) 2,319,000</td>
      <td>(R) 2,343,000</td>
      <td>(R) 2,455,000</td>
      <td>(R) 3,062,000</td>
      <td>2,745,000</td>
      <td>2,710,000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Crashes</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>N</td>
      <td>6,471,000</td>
      <td>6,117,000</td>
      <td>6,000,000</td>
      <td>...</td>
      <td>5,505,000</td>
      <td>5,419,000</td>
      <td>5,338,000</td>
      <td>5,615,000</td>
      <td>5,687,000</td>
      <td>6,064,000</td>
      <td>6,296,000</td>
      <td>6,821,000</td>
      <td>6,453,000</td>
      <td>6,734,000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Vehicle-miles (millions)</td>
      <td>718,763</td>
      <td>887,811</td>
      <td>1,109,724</td>
      <td>1,327,664</td>
      <td>1,527,295</td>
      <td>1,774,826</td>
      <td>2,144,362</td>
      <td>2,172,050</td>
      <td>2,247,151</td>
      <td>...</td>
      <td>2,956,764</td>
      <td>2,967,266</td>
      <td>2,950,402</td>
      <td>2,969,433</td>
      <td>2,988,280</td>
      <td>3,025,656</td>
      <td>3,095,373</td>
      <td>3,174,408</td>
      <td>3,212,347</td>
      <td>3,240,327</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 36 columns</p>
</div>




```python
cars_data.set_index(cars_data.columns[0], inplace=True)
cars_data = cars_data.T
cars_data.head()
```


```python
cars_data = cars_data.replace(',','', regex=True)
cars_data['Injured persons'] = cars_data['Injured persons'].str.replace(r"\(.*\)","")
cars_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>nan</th>
      <th>Fatalities</th>
      <th>Injured persons</th>
      <th>Crashes</th>
      <th>Vehicle-miles (millions)</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>36399</td>
      <td>N</td>
      <td>N</td>
      <td>718763</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>47089</td>
      <td>N</td>
      <td>N</td>
      <td>887811</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>52627</td>
      <td>N</td>
      <td>N</td>
      <td>1109724</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>44525</td>
      <td>N</td>
      <td>N</td>
      <td>1327664</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>51091</td>
      <td>N</td>
      <td>N</td>
      <td>1527295</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>43825</td>
      <td>N</td>
      <td>N</td>
      <td>1774826</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>44599</td>
      <td>3246000</td>
      <td>6471000</td>
      <td>2144362</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>41508</td>
      <td>3107000</td>
      <td>6117000</td>
      <td>2172050</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>39250</td>
      <td>3079000</td>
      <td>6000000</td>
      <td>2247151</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>40150</td>
      <td>3163000</td>
      <td>6106000</td>
      <td>2296378</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove rows with missing data
cars_data = cars_data[6:]
cars_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>nan</th>
      <th>Fatalities</th>
      <th>Injured persons</th>
      <th>Crashes</th>
      <th>Vehicle-miles (millions)</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990</th>
      <td>44599</td>
      <td>3246000</td>
      <td>6471000</td>
      <td>2144362</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>41508</td>
      <td>3107000</td>
      <td>6117000</td>
      <td>2172050</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>39250</td>
      <td>3079000</td>
      <td>6000000</td>
      <td>2247151</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>40150</td>
      <td>3163000</td>
      <td>6106000</td>
      <td>2296378</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>40716</td>
      <td>3275000</td>
      <td>6496000</td>
      <td>2357588</td>
    </tr>
  </tbody>
</table>
</div>




```python
cars_data['Vehicle-miles (millions)'] = cars_data['Vehicle-miles (millions)'].astype(int)
cars_data['Fatalities'] = cars_data['Fatalities'].astype(int)
cars_data['Injured persons'] = cars_data['Injured persons'].astype(int)
```


```python
cars_data['car_fatality_per_mil_mile'] = cars_data['Fatalities'] / cars_data['Vehicle-miles (millions)']
cars_data['car_injury_per_mil_mile'] = cars_data['Injured persons'] / cars_data['Vehicle-miles (millions)']
cars_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>nan</th>
      <th>Fatalities</th>
      <th>Injured persons</th>
      <th>Crashes</th>
      <th>Vehicle-miles (millions)</th>
      <th>car_fatality_per_mil_mile</th>
      <th>car_injury_per_mil_mile</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990</th>
      <td>44599</td>
      <td>3246000</td>
      <td>6471000</td>
      <td>2144362</td>
      <td>0.020798</td>
      <td>1.513737</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>41508</td>
      <td>3107000</td>
      <td>6117000</td>
      <td>2172050</td>
      <td>0.019110</td>
      <td>1.430446</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>39250</td>
      <td>3079000</td>
      <td>6000000</td>
      <td>2247151</td>
      <td>0.017467</td>
      <td>1.370179</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>40150</td>
      <td>3163000</td>
      <td>6106000</td>
      <td>2296378</td>
      <td>0.017484</td>
      <td>1.377386</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>40716</td>
      <td>3275000</td>
      <td>6496000</td>
      <td>2357588</td>
      <td>0.017270</td>
      <td>1.389132</td>
    </tr>
  </tbody>
</table>
</div>




```python
car_mile = cars_data[['car_fatality_per_mil_mile', 'car_injury_per_mil_mile']]
```


```python
fly_data = pd.read_csv('better_airplane_data.csv') 
fly_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Table 2-9:  U.S. Air Carriera Safety Data</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>...</th>
      <th>Unnamed: 26</th>
      <th>Unnamed: 27</th>
      <th>Unnamed: 28</th>
      <th>Unnamed: 29</th>
      <th>Unnamed: 30</th>
      <th>Unnamed: 31</th>
      <th>Unnamed: 32</th>
      <th>Unnamed: 33</th>
      <th>Unnamed: 34</th>
      <th>Unnamed: 35</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1960</td>
      <td>1965</td>
      <td>1970</td>
      <td>1975</td>
      <td>1980</td>
      <td>1985</td>
      <td>1990</td>
      <td>1991</td>
      <td>1992</td>
      <td>...</td>
      <td>2009</td>
      <td>2010</td>
      <td>2011</td>
      <td>2012</td>
      <td>2013</td>
      <td>(R) 2014</td>
      <td>(R) 2015</td>
      <td>(R) 2016</td>
      <td>(R) 2017</td>
      <td>(P) 2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Total fatalities</td>
      <td>499</td>
      <td>261</td>
      <td>146</td>
      <td>124</td>
      <td>1</td>
      <td>526</td>
      <td>39</td>
      <td>50</td>
      <td>33</td>
      <td>...</td>
      <td>52</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Total seriously injured persons</td>
      <td>N</td>
      <td>N</td>
      <td>107</td>
      <td>81</td>
      <td>19</td>
      <td>30</td>
      <td>29</td>
      <td>26</td>
      <td>22</td>
      <td>...</td>
      <td>26</td>
      <td>17</td>
      <td>21</td>
      <td>18</td>
      <td>9</td>
      <td>14</td>
      <td>24</td>
      <td>18</td>
      <td>19</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Total accidents</td>
      <td>90</td>
      <td>83</td>
      <td>55</td>
      <td>37</td>
      <td>19</td>
      <td>21</td>
      <td>24</td>
      <td>26</td>
      <td>18</td>
      <td>...</td>
      <td>30</td>
      <td>30</td>
      <td>33</td>
      <td>26</td>
      <td>23</td>
      <td>31</td>
      <td>29</td>
      <td>30</td>
      <td>32</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fatal accidents</td>
      <td>17</td>
      <td>9</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
fly_data.columns = fly_data.iloc[0]
fly_data = fly_data[1:6]
fly_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NaN</th>
      <th>1960</th>
      <th>1965</th>
      <th>1970</th>
      <th>1975</th>
      <th>1980</th>
      <th>1985</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>...</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>(R) 2014</th>
      <th>(R) 2015</th>
      <th>(R) 2016</th>
      <th>(R) 2017</th>
      <th>(P) 2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Total fatalities</td>
      <td>499</td>
      <td>261</td>
      <td>146</td>
      <td>124</td>
      <td>1</td>
      <td>526</td>
      <td>39</td>
      <td>50</td>
      <td>33</td>
      <td>...</td>
      <td>52</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Total seriously injured persons</td>
      <td>N</td>
      <td>N</td>
      <td>107</td>
      <td>81</td>
      <td>19</td>
      <td>30</td>
      <td>29</td>
      <td>26</td>
      <td>22</td>
      <td>...</td>
      <td>26</td>
      <td>17</td>
      <td>21</td>
      <td>18</td>
      <td>9</td>
      <td>14</td>
      <td>24</td>
      <td>18</td>
      <td>19</td>
      <td>25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Total accidents</td>
      <td>90</td>
      <td>83</td>
      <td>55</td>
      <td>37</td>
      <td>19</td>
      <td>21</td>
      <td>24</td>
      <td>26</td>
      <td>18</td>
      <td>...</td>
      <td>30</td>
      <td>30</td>
      <td>33</td>
      <td>26</td>
      <td>23</td>
      <td>31</td>
      <td>29</td>
      <td>30</td>
      <td>32</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fatal accidents</td>
      <td>17</td>
      <td>9</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Aircraft-miles (millions)</td>
      <td>1,130</td>
      <td>1,536</td>
      <td>2,685</td>
      <td>2,478</td>
      <td>2,924</td>
      <td>3,631</td>
      <td>4,948</td>
      <td>4,825</td>
      <td>5,039</td>
      <td>...</td>
      <td>7,466</td>
      <td>7,598</td>
      <td>7,714</td>
      <td>7,660</td>
      <td>7,673</td>
      <td>7,691</td>
      <td>7,822</td>
      <td>8,017</td>
      <td>8,155</td>
      <td>8,474</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
fly_data.set_index(fly_data.columns[0], inplace=True)
fly_data = fly_data.T
fly_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>nan</th>
      <th>Total fatalities</th>
      <th>Total seriously injured persons</th>
      <th>Total accidents</th>
      <th>Fatal accidents</th>
      <th>Aircraft-miles (millions)</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>499</td>
      <td>N</td>
      <td>90</td>
      <td>17</td>
      <td>1,130</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>261</td>
      <td>N</td>
      <td>83</td>
      <td>9</td>
      <td>1,536</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>146</td>
      <td>107</td>
      <td>55</td>
      <td>8</td>
      <td>2,685</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>124</td>
      <td>81</td>
      <td>37</td>
      <td>3</td>
      <td>2,478</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>1</td>
      <td>19</td>
      <td>19</td>
      <td>1</td>
      <td>2,924</td>
    </tr>
  </tbody>
</table>
</div>




```python
fly_data = fly_data.replace(',','', regex=True)
fly_data['Total fatalities'] = fly_data['Total fatalities'].str.replace(r"\(.*\)","")
fly_data['Total seriously injured persons'] = fly_data['Total seriously injured persons'].str.replace(r"\(.*\)","")

fly_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>nan</th>
      <th>Total fatalities</th>
      <th>Total seriously injured persons</th>
      <th>Total accidents</th>
      <th>Fatal accidents</th>
      <th>Aircraft-miles (millions)</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>499</td>
      <td>N</td>
      <td>90</td>
      <td>17</td>
      <td>1130</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>261</td>
      <td>N</td>
      <td>83</td>
      <td>9</td>
      <td>1536</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>146</td>
      <td>107</td>
      <td>55</td>
      <td>8</td>
      <td>2685</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>124</td>
      <td>81</td>
      <td>37</td>
      <td>3</td>
      <td>2478</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>1</td>
      <td>19</td>
      <td>19</td>
      <td>1</td>
      <td>2924</td>
    </tr>
  </tbody>
</table>
</div>




```python
fly_data = fly_data[6:]
fly_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>nan</th>
      <th>Total fatalities</th>
      <th>Total seriously injured persons</th>
      <th>Total accidents</th>
      <th>Fatal accidents</th>
      <th>Aircraft-miles (millions)</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990</th>
      <td>39</td>
      <td>29</td>
      <td>24</td>
      <td>6</td>
      <td>4948</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>50</td>
      <td>26</td>
      <td>26</td>
      <td>4</td>
      <td>4825</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>33</td>
      <td>22</td>
      <td>18</td>
      <td>4</td>
      <td>5039</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>1</td>
      <td>19</td>
      <td>23</td>
      <td>1</td>
      <td>5249</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>239</td>
      <td>31</td>
      <td>23</td>
      <td>4</td>
      <td>5478</td>
    </tr>
  </tbody>
</table>
</div>




```python
fly_data['Fatal accidents'] = fly_data['Fatal accidents'].astype(int)
fly_data['Aircraft-miles (millions)'] = fly_data['Aircraft-miles (millions)'].astype(int)
fly_data['Total accidents'] = fly_data['Total accidents'].astype(int)
fly_data.to_csv('fly_data.csv', index=True)
```


```python
fly_data['fatal_per_mil_mile'] = fly_data['Fatal accidents'] / fly_data['Aircraft-miles (millions)']
fly_data['injury_per_mil_mile'] = fly_data['Total accidents'] / fly_data['Aircraft-miles (millions)']
```


```python
air_mile = fly_data[['fatal_per_mil_mile', 'injury_per_mil_mile']]
```


```python
both_mile = car_mile.merge(air_mile, left_index=True, right_index=True)
```


```python
both_mile.to_csv('by_mile.csv', index=True)
```


```python
total_fly_fatal = fly_data[['Total fatalities', 'Total seriously injured persons']]
total_car_fatal = cars_data[['Fatalities', 'Injured persons']]
```


```python
total_fatal = total_fly_fatal.merge(total_car_fatal, left_index=True, right_index=True)
```


```python
total_fatal.to_csv('total_fatal.csv', index=True)
```
