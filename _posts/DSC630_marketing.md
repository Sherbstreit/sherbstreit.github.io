```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv('dodgers.csv')
df
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
      <th>month</th>
      <th>day</th>
      <th>attend</th>
      <th>day_of_week</th>
      <th>opponent</th>
      <th>temp</th>
      <th>skies</th>
      <th>day_night</th>
      <th>cap</th>
      <th>shirt</th>
      <th>fireworks</th>
      <th>bobblehead</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>APR</td>
      <td>10</td>
      <td>56000</td>
      <td>Tuesday</td>
      <td>Pirates</td>
      <td>67</td>
      <td>Clear</td>
      <td>Day</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>APR</td>
      <td>11</td>
      <td>29729</td>
      <td>Wednesday</td>
      <td>Pirates</td>
      <td>58</td>
      <td>Cloudy</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>APR</td>
      <td>12</td>
      <td>28328</td>
      <td>Thursday</td>
      <td>Pirates</td>
      <td>57</td>
      <td>Cloudy</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>APR</td>
      <td>13</td>
      <td>31601</td>
      <td>Friday</td>
      <td>Padres</td>
      <td>54</td>
      <td>Cloudy</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>YES</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>APR</td>
      <td>14</td>
      <td>46549</td>
      <td>Saturday</td>
      <td>Padres</td>
      <td>57</td>
      <td>Cloudy</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>76</th>
      <td>SEP</td>
      <td>29</td>
      <td>40724</td>
      <td>Saturday</td>
      <td>Rockies</td>
      <td>84</td>
      <td>Cloudy</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>77</th>
      <td>SEP</td>
      <td>30</td>
      <td>35607</td>
      <td>Sunday</td>
      <td>Rockies</td>
      <td>95</td>
      <td>Clear</td>
      <td>Day</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>78</th>
      <td>OCT</td>
      <td>1</td>
      <td>33624</td>
      <td>Monday</td>
      <td>Giants</td>
      <td>86</td>
      <td>Clear</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>79</th>
      <td>OCT</td>
      <td>2</td>
      <td>42473</td>
      <td>Tuesday</td>
      <td>Giants</td>
      <td>83</td>
      <td>Clear</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>80</th>
      <td>OCT</td>
      <td>3</td>
      <td>34014</td>
      <td>Wednesday</td>
      <td>Giants</td>
      <td>82</td>
      <td>Cloudy</td>
      <td>Night</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
<p>81 rows × 12 columns</p>
</div>




```python
# Problem given specified night games
df = df[df['day_night'] == 'Night' ]
```


```python
# Check attendance means by day and month
df.groupby(['day_of_week', 'month'], as_index=True)[['attend']].mean()
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
      <th></th>
      <th>attend</th>
    </tr>
    <tr>
      <th>day_of_week</th>
      <th>month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">Friday</th>
      <th>APR</th>
      <td>38204.000000</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>40321.333333</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>43873.000000</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>45097.500000</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>37593.333333</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>38650.000000</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Monday</th>
      <th>APR</th>
      <td>26376.000000</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>34768.500000</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>33303.666667</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>50559.000000</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>35347.000000</td>
    </tr>
    <tr>
      <th>OCT</th>
      <td>33624.000000</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>33540.000000</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Saturday</th>
      <th>APR</th>
      <td>50395.500000</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>43436.000000</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>54014.000000</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>45210.000000</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>36559.666667</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>39721.666667</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sunday</th>
      <th>JUL</th>
      <td>55359.000000</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>44005.000000</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Thursday</th>
      <th>APR</th>
      <td>28328.000000</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>54621.000000</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>49006.000000</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>26773.000000</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>43309.000000</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Tuesday</th>
      <th>APR</th>
      <td>44014.000000</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>55512.000000</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>46738.000000</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>55279.000000</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>43671.000000</td>
    </tr>
    <tr>
      <th>OCT</th>
      <td>42473.000000</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>40619.000000</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Wednesday</th>
      <th>APR</th>
      <td>28037.000000</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>38628.500000</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>53570.000000</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>43494.000000</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>29751.000000</td>
    </tr>
    <tr>
      <th>OCT</th>
      <td>34014.000000</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>50560.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check number of games occuring
df.groupby(['day_of_week', 'month'], as_index=True)[['attend']].count()
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
      <th></th>
      <th>attend</th>
    </tr>
    <tr>
      <th>day_of_week</th>
      <th>month</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">Friday</th>
      <th>APR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>3</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>1</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>2</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>3</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Monday</th>
      <th>APR</th>
      <td>1</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>2</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>3</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>3</td>
    </tr>
    <tr>
      <th>OCT</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Saturday</th>
      <th>APR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>2</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>1</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>3</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Sunday</th>
      <th>JUL</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Thursday</th>
      <th>APR</th>
      <td>1</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>1</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Tuesday</th>
      <th>APR</th>
      <td>1</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>2</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>3</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>3</td>
    </tr>
    <tr>
      <th>OCT</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>1</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Wednesday</th>
      <th>APR</th>
      <td>2</td>
    </tr>
    <tr>
      <th>AUG</th>
      <td>2</td>
    </tr>
    <tr>
      <th>JUL</th>
      <td>1</td>
    </tr>
    <tr>
      <th>JUN</th>
      <td>1</td>
    </tr>
    <tr>
      <th>MAY</th>
      <td>2</td>
    </tr>
    <tr>
      <th>OCT</th>
      <td>1</td>
    </tr>
    <tr>
      <th>SEP</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# low value count - not keeping for analysis
df['cap'].value_counts()
```




    NO     65
    YES     1
    Name: cap, dtype: int64




```python
# low value count - not keeping for analysis
df['shirt'].value_counts()
```




    NO     64
    YES     2
    Name: shirt, dtype: int64




```python
# visualize attendance by day of the week
sns.set(font_scale=1.3)
plt.rcParams['figure.figsize']=(15,8)
ax = sns.boxplot(x="day_of_week", y="attend", hue="skies",
                 data=df, palette="Set3")
ax.set_title('Attendance by Day of the Week')
```




    Text(0.5, 1.0, 'Attendance by Day of the Week')




    
![png](output_7_1.png)
    



```python
# attendance by month
ax = sns.boxplot(x="month", y="attend",
                 data=df, palette="Set3")
ax.set_title('Attendance by Month')
```




    Text(0.5, 1.0, 'Attendance by Month')




    
![png](output_8_1.png)
    



```python
# Attendance by opponent
plt.rcParams['figure.figsize']=(18,8)
ax = sns.boxplot(x="opponent", y="attend",
                 data=df, palette="Set3")
ax.set_title('Attendance by Opponent')
```




    Text(0.5, 1.0, 'Attendance by Opponent')




    
![png](output_9_1.png)
    



```python
# attendance by fireworks offered
plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x='fireworks', y="attend",
                 data=df, palette="Set3")
ax.set_title('Attendance on Nights with Fireworks')
```




    Text(0.5, 1.0, 'Attendance on Nights with Fireworks')




    
![png](output_10_1.png)
    



```python
# attendance by bobblehead offered
ax = sns.boxplot(x='bobblehead', y="attend",
                 data=df, palette="Set3")
ax.set_title('Attendance on Nights with Bobbleheads Offered')
```




    Text(0.5, 1.0, 'Attendance on Nights with Bobbleheads Offered')




    
![png](output_11_1.png)
    



```python
ax = sns.lmplot( x='temp', y='attend', data=df, legend=False)
ax.fig.suptitle('Attendance by Temperature Outside',
                  fontsize=14)
```




    Text(0.5, 0.98, 'Attendance by Temperature Outside')




    
![png](output_12_1.png)
    



```python
ax = sns.lmplot( x="day", y="attend", data=df, legend=False)
ax.fig.suptitle('Attendance by Day of the Month',
                  fontsize=14)
```




    Text(0.5, 0.98, 'Attendance by Day of the Month')




    
![png](output_13_1.png)
    



```python
# Choose days of week and month as independent variables
X = df[['day_of_week', 'month', 'bobblehead']]
```


```python
# convert to dummies
X = pd.get_dummies(X, drop_first=True)
X
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
      <th>day_of_week_Monday</th>
      <th>day_of_week_Saturday</th>
      <th>day_of_week_Sunday</th>
      <th>day_of_week_Thursday</th>
      <th>day_of_week_Tuesday</th>
      <th>day_of_week_Wednesday</th>
      <th>month_AUG</th>
      <th>month_JUL</th>
      <th>month_JUN</th>
      <th>month_MAY</th>
      <th>month_OCT</th>
      <th>month_SEP</th>
      <th>bobblehead_YES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 13 columns</p>
</div>




```python
# target variable
Y = df[['attend']]
Y
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
      <th>attend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>29729</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28328</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31601</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46549</td>
    </tr>
    <tr>
      <th>6</th>
      <td>26376</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>75</th>
      <td>37133</td>
    </tr>
    <tr>
      <th>76</th>
      <td>40724</td>
    </tr>
    <tr>
      <th>78</th>
      <td>33624</td>
    </tr>
    <tr>
      <th>79</th>
      <td>42473</td>
    </tr>
    <tr>
      <th>80</th>
      <td>34014</td>
    </tr>
  </tbody>
</table>
<p>66 rows × 1 columns</p>
</div>




```python
import statsmodels.regression.linear_model as lm
```


```python
# fit OLS model and view results
regressor_ols = lm.OLS(endog = Y, exog = X).fit()
regressor_ols.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>attend</td>      <th>  R-squared (uncentered):</th>      <td>   0.917</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.897</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   45.05</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 20 Dec 2020</td> <th>  Prob (F-statistic):</th>          <td>6.27e-24</td>
</tr>
<tr>
  <th>Time:</th>                 <td>21:33:26</td>     <th>  Log-Likelihood:    </th>          <td> -713.70</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    66</td>      <th>  AIC:               </th>          <td>   1453.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    53</td>      <th>  BIC:               </th>          <td>   1482.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    13</td>      <th>                     </th>              <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>day_of_week_Monday</th>    <td> 1.029e+04</td> <td> 5004.624</td> <td>    2.057</td> <td> 0.045</td> <td>  255.996</td> <td> 2.03e+04</td>
</tr>
<tr>
  <th>day_of_week_Saturday</th>  <td> 1.847e+04</td> <td> 4876.466</td> <td>    3.788</td> <td> 0.000</td> <td> 8693.249</td> <td> 2.83e+04</td>
</tr>
<tr>
  <th>day_of_week_Sunday</th>    <td> 1.912e+04</td> <td> 1.07e+04</td> <td>    1.789</td> <td> 0.079</td> <td>-2318.236</td> <td> 4.06e+04</td>
</tr>
<tr>
  <th>day_of_week_Thursday</th>  <td> 1.348e+04</td> <td> 6936.344</td> <td>    1.943</td> <td> 0.057</td> <td> -435.953</td> <td> 2.74e+04</td>
</tr>
<tr>
  <th>day_of_week_Tuesday</th>   <td> 1.732e+04</td> <td> 5704.671</td> <td>    3.036</td> <td> 0.004</td> <td> 5875.281</td> <td> 2.88e+04</td>
</tr>
<tr>
  <th>day_of_week_Wednesday</th> <td> 1.574e+04</td> <td> 5033.821</td> <td>    3.128</td> <td> 0.003</td> <td> 5647.106</td> <td> 2.58e+04</td>
</tr>
<tr>
  <th>month_AUG</th>             <td>  2.94e+04</td> <td> 4684.860</td> <td>    6.275</td> <td> 0.000</td> <td>    2e+04</td> <td> 3.88e+04</td>
</tr>
<tr>
  <th>month_JUL</th>             <td> 2.804e+04</td> <td> 5478.384</td> <td>    5.118</td> <td> 0.000</td> <td>  1.7e+04</td> <td>  3.9e+04</td>
</tr>
<tr>
  <th>month_JUN</th>             <td> 3.403e+04</td> <td> 5690.131</td> <td>    5.980</td> <td> 0.000</td> <td> 2.26e+04</td> <td> 4.54e+04</td>
</tr>
<tr>
  <th>month_MAY</th>             <td> 2.295e+04</td> <td> 4402.888</td> <td>    5.212</td> <td> 0.000</td> <td> 1.41e+04</td> <td> 3.18e+04</td>
</tr>
<tr>
  <th>month_OCT</th>             <td> 2.225e+04</td> <td> 8667.153</td> <td>    2.567</td> <td> 0.013</td> <td> 4867.867</td> <td> 3.96e+04</td>
</tr>
<tr>
  <th>month_SEP</th>             <td> 2.803e+04</td> <td> 5308.575</td> <td>    5.279</td> <td> 0.000</td> <td> 1.74e+04</td> <td> 3.87e+04</td>
</tr>
<tr>
  <th>bobblehead_YES</th>        <td> 1.013e+04</td> <td> 5630.124</td> <td>    1.799</td> <td> 0.078</td> <td>-1164.735</td> <td> 2.14e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16.930</td> <th>  Durbin-Watson:     </th> <td>   1.154</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  20.017</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.171</td> <th>  Prob(JB):          </th> <td>4.50e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.341</td> <th>  Cond. No.          </th> <td>    4.78</td>
</tr>
</table><br/><br/>Notes:<br/>[1] R² is computed without centering (uncentered) since the model does not contain a constant.<br/>[2] Standard Errors assume that the covariance matrix of the errors is correctly specified.



## Marketing should be done on Mondays in May to get the highest increase in attendance. 

### Monday was chosen because it has the lowest overall attendance. October had the lowest coefficient, but was not chosen because there was only 1 Monday game. This would not result in a significant increase. May was chosen because it had the second highest coefficient, with 3 Monday games.  In addition, offering Bobbleheads can further increase attendance.


```python

```
