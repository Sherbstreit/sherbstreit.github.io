---
title: "Predicting Home Sale Price"
date: 2021-06-05
tags: [data wrangling, data cleaning, regression, python, predictive analytics, model tuning, visualization]
header:
  image: images/regression.jpeg
excerpt: "Using Regression to predict home sale price"
mathjax: "true"
---

## Overview of Findings

<iframe src="https://bellevueuniversity-my.sharepoint.com/:p:/g/personal/sherbstreit_my365_bellevue_edu/EU5jPU6ZYhVCpSlJpq-kNKYBwFRhi3vAXB5AHPOfBdMM3A?e=Ftq0CS&amp;action=embedview&amp;wdAr=1.7777777777777777" width="962px" height="565px" frameborder="0">This is an embedded <a target="_blank" href="https://office.com">Microsoft Office</a> presentation, powered by <a target="_blank" href="https://office.com/webapps">Office</a>.</iframe>


<br />
# The Code
## Using 7 Regression Models to Predict House Sales Price


```python
import pandas as pd
import string
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
```


```python
train = pd.read_csv('home_prices_train.csv')
test = pd.read_csv('home_prices_test.csv')
print('The training set contains {} data points. The test set contains {} data points'.format(len(train), len(test)))
```

    The training set contains 1460 data points. The test set contains 1459 data points



```python
pd.set_option('display.max_columns', 100)
train.head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.describe()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1379.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>46.549315</td>
      <td>567.240411</td>
      <td>1057.429452</td>
      <td>1162.626712</td>
      <td>346.992466</td>
      <td>5.844521</td>
      <td>1515.463699</td>
      <td>0.425342</td>
      <td>0.057534</td>
      <td>1.565068</td>
      <td>0.382877</td>
      <td>2.866438</td>
      <td>1.046575</td>
      <td>6.517808</td>
      <td>0.613014</td>
      <td>1978.506164</td>
      <td>1.767123</td>
      <td>472.980137</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>161.319273</td>
      <td>441.866955</td>
      <td>438.705324</td>
      <td>386.587738</td>
      <td>436.528436</td>
      <td>48.623081</td>
      <td>525.480383</td>
      <td>0.518911</td>
      <td>0.238753</td>
      <td>0.550916</td>
      <td>0.502885</td>
      <td>0.815778</td>
      <td>0.220338</td>
      <td>1.625393</td>
      <td>0.644666</td>
      <td>24.689725</td>
      <td>0.747315</td>
      <td>213.804841</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>334.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>223.000000</td>
      <td>795.750000</td>
      <td>882.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1129.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>1961.000000</td>
      <td>1.000000</td>
      <td>334.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>0.000000</td>
      <td>477.500000</td>
      <td>991.500000</td>
      <td>1087.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1464.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>1980.000000</td>
      <td>2.000000</td>
      <td>480.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>0.000000</td>
      <td>808.000000</td>
      <td>1298.250000</td>
      <td>1391.250000</td>
      <td>728.000000</td>
      <td>0.000000</td>
      <td>1776.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>2002.000000</td>
      <td>2.000000</td>
      <td>576.000000</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>2336.000000</td>
      <td>6110.000000</td>
      <td>4692.000000</td>
      <td>2065.000000</td>
      <td>572.000000</td>
      <td>5642.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>14.000000</td>
      <td>3.000000</td>
      <td>2010.000000</td>
      <td>4.000000</td>
      <td>1418.000000</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize sales price to overall house quality
ax = sns.boxplot(x='OverallQual', y='SalePrice', data=train, palette='YlGnBu')
ax.set(xlabel='Overall House Quality', ylabel='Sale Price')
```




    [Text(0.5, 0, 'Overall House Quality'), Text(0, 0.5, 'Sale Price')]




<img src="{{ site.url }}{{ site.baseurl }}/images/output_5_1.png" alt="linearly separable data">
   



```python
# Visualize sales price to house condition
ax = sns.boxplot(x='OverallCond', y='SalePrice', data=train, palette='YlGnBu')
ax.set(xlabel='Overall House Condition', ylabel='Sale Price')
```




    [Text(0.5, 0, 'Overall House Condition'), Text(0, 0.5, 'Sale Price')]




<img src="{{ site.url }}{{ site.baseurl }}/images/output_6_1.png" alt="linearly separable data">
   

    



```python
# Visualize sales price to Bedrooms
ax = sns.boxplot(x='BedroomAbvGr', y='SalePrice', data=train, palette='YlGnBu')
ax.set(xlabel='Number Bedrooms', ylabel='Sale Price')
```




    [Text(0.5, 0, 'Number Bedrooms'), Text(0, 0.5, 'Sale Price')]




<img src="{{ site.url }}{{ site.baseurl }}/images/output_7_1.png" alt="linearly separable data">
   

    



```python
# Visualize sales price to full bathrooms
ax = sns.boxplot(x='FullBath', y='SalePrice', data=train, palette='YlGnBu')
ax.set(xlabel='Number Full Bathrooms', ylabel='Sale Price')
```




    [Text(0.5, 0, 'Number Full Bathrooms'), Text(0, 0.5, 'Sale Price')]




<img src="{{ site.url }}{{ site.baseurl }}/images/output_8_1.png" alt="linearly separable data">
    

    



```python
# Visualize sales price to overall kitchen quality
ax = sns.boxplot(x='KitchenQual', y='SalePrice', data=train, palette='YlGnBu',
                order=['Fa', 'TA', 'Gd', 'Ex'])
ax.set(xlabel='Kitchen Quality', ylabel='Sale Price')
```




    [Text(0.5, 0, 'Kitchen Quality'), Text(0, 0.5, 'Sale Price')]




<img src="{{ site.url }}{{ site.baseurl }}/images/output_9_1.png" alt="linearly separable data">
  
    



```python
# Visualize sales price to number fireplaces
ax = sns.boxplot(x='Fireplaces', y='SalePrice', data=train, palette='YlGnBu')
ax.set(xlabel='Number of Fireplaces', ylabel='Sale Price')
```




    [Text(0.5, 0, 'Number of Fireplaces'), Text(0, 0.5, 'Sale Price')]




<img src="{{ site.url }}{{ site.baseurl }}/images/output_10_1.png" alt="linearly separable data">
    
    



```python
# Visualize sales price to number cars fit in garage
ax = sns.boxplot(x='GarageCars', y='SalePrice', data=train, palette='YlGnBu')
ax.set(xlabel='Number Car Garage', ylabel='Sale Price')
```




    [Text(0.5, 0, 'Number Car Garage'), Text(0, 0.5, 'Sale Price')]




<img src="{{ site.url }}{{ site.baseurl }}/images/output_11_1.png" alt="linearly separable data">
    




```python
train[train['GarageCars']==4]
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>420</th>
      <td>421</td>
      <td>90</td>
      <td>RM</td>
      <td>78.0</td>
      <td>7060</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>SFoyer</td>
      <td>7</td>
      <td>5</td>
      <td>1997</td>
      <td>1998</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>200.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>1309</td>
      <td>Unf</td>
      <td>0</td>
      <td>35</td>
      <td>1344</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1344</td>
      <td>0</td>
      <td>0</td>
      <td>1344</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>1997.0</td>
      <td>Fin</td>
      <td>4</td>
      <td>784</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
      <td>WD</td>
      <td>Alloca</td>
      <td>206300</td>
    </tr>
    <tr>
      <th>747</th>
      <td>748</td>
      <td>70</td>
      <td>RM</td>
      <td>65.0</td>
      <td>11700</td>
      <td>Pave</td>
      <td>Pave</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>OldTown</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>7</td>
      <td>1880</td>
      <td>2003</td>
      <td>Mansard</td>
      <td>CompShg</td>
      <td>Stucco</td>
      <td>Stucco</td>
      <td>None</td>
      <td>0.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Stone</td>
      <td>TA</td>
      <td>Fa</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>1240</td>
      <td>1240</td>
      <td>GasW</td>
      <td>TA</td>
      <td>N</td>
      <td>SBrkr</td>
      <td>1320</td>
      <td>1320</td>
      <td>0</td>
      <td>2640</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1950.0</td>
      <td>Unf</td>
      <td>4</td>
      <td>864</td>
      <td>TA</td>
      <td>TA</td>
      <td>N</td>
      <td>181</td>
      <td>0</td>
      <td>386</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>265979</td>
    </tr>
    <tr>
      <th>1190</th>
      <td>1191</td>
      <td>190</td>
      <td>RL</td>
      <td>NaN</td>
      <td>32463</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Mod</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>2fmCon</td>
      <td>1Story</td>
      <td>4</td>
      <td>4</td>
      <td>1961</td>
      <td>1975</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>Stone</td>
      <td>149.0</td>
      <td>TA</td>
      <td>Gd</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>Av</td>
      <td>BLQ</td>
      <td>1159</td>
      <td>Unf</td>
      <td>0</td>
      <td>90</td>
      <td>1249</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1622</td>
      <td>0</td>
      <td>0</td>
      <td>1622</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>2Types</td>
      <td>1975.0</td>
      <td>Fin</td>
      <td>4</td>
      <td>1356</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>439</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>168000</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>1341</td>
      <td>20</td>
      <td>RL</td>
      <td>70.0</td>
      <td>8294</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>5</td>
      <td>1971</td>
      <td>1971</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Unf</td>
      <td>0</td>
      <td>Unf</td>
      <td>0</td>
      <td>858</td>
      <td>858</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>872</td>
      <td>0</td>
      <td>0</td>
      <td>872</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Detchd</td>
      <td>1974.0</td>
      <td>Unf</td>
      <td>4</td>
      <td>480</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdWo</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>123000</td>
    </tr>
    <tr>
      <th>1350</th>
      <td>1351</td>
      <td>90</td>
      <td>RL</td>
      <td>91.0</td>
      <td>11643</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Artery</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>1969</td>
      <td>1969</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>BrkFace</td>
      <td>368.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>LwQ</td>
      <td>500</td>
      <td>Unf</td>
      <td>0</td>
      <td>748</td>
      <td>1248</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1338</td>
      <td>1296</td>
      <td>0</td>
      <td>2634</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>TA</td>
      <td>12</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Detchd</td>
      <td>1969.0</td>
      <td>Unf</td>
      <td>4</td>
      <td>968</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>200000</td>
    </tr>
  </tbody>
</table>
</div>



## Handling null values


```python
# Output too long
# pd.set_option('display.max_rows', 200)
# train.isna().sum()
```


```python
train['MSZoning'].unique()
```




    array(['RL', 'RM', 'C (all)', 'FV', 'RH'], dtype=object)




```python
# drop Id because irrelevant
# drop Pool quality because only 7 not NA, and Pool area tells us if there is a pool
# drop all non-residential zoning
# garage cars and garage size in sqft is redundant
train = train.drop(train[train.MSZoning.isin(['FV', 'C (all)'])].index)
train.drop(['Id', 'PoolQC', 'GarageCars'], axis=1, inplace=True)
test.drop(['Id', 'PoolQC', 'GarageCars'], axis=1, inplace=True)
```


```python
# slab foundations have no basement, fill nan
train.BsmtQual=np.where(train.Foundation=='Slab',train.BsmtQual.fillna('None'),train.BsmtQual)
train.BsmtCond=np.where(train.Foundation=='Slab',train.BsmtCond.fillna('None'),train.BsmtCond)
train.BsmtExposure=np.where(train.Foundation=='Slab',train.BsmtExposure.fillna('None'),train.BsmtExposure)
train.BsmtFinType1=np.where(train.Foundation=='Slab',train.BsmtFinType1.fillna('None'),train.BsmtFinType1)
train.BsmtFinType2=np.where(train.Foundation=='Slab',train.BsmtFinType2.fillna('None'),train.BsmtFinType2)
# per data source, nan in these columns means condition does not exist
train.Fence.fillna('None', inplace=True)
train.FireplaceQu.fillna('None', inplace=True)
train.GarageType.fillna('None', inplace=True)
train.GarageFinish.fillna('None', inplace=True)
train.GarageQual.fillna('None', inplace=True)
train.GarageCond.fillna('None', inplace=True)
train.Alley.fillna('No_access', inplace=True)
train.MiscFeature.fillna('None', inplace=True)
# Likely no garage, but fill nan with mean year bc 0 would skew data
# fill nan to avoid removing rows or column
train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean(), inplace=True)

# perform same nan handling to test set
test.BsmtQual=np.where(test.Foundation=='Slab',test.BsmtQual.fillna('None'),test.BsmtQual)
test.BsmtCond=np.where(test.Foundation=='Slab',test.BsmtCond.fillna('None'),test.BsmtCond)
test.BsmtExposure=np.where(test.Foundation=='Slab',test.BsmtExposure.fillna('None'),test.BsmtExposure)
test.BsmtFinType1=np.where(test.Foundation=='Slab',test.BsmtFinType1.fillna('None'),test.BsmtFinType1)
test.BsmtFinType2=np.where(test.Foundation=='Slab',test.BsmtFinType2.fillna('None'),test.BsmtFinType2)
test.Fence.fillna('None', inplace=True)
test.FireplaceQu.fillna('None', inplace=True)
test.GarageType.fillna('None', inplace=True)
test.GarageFinish.fillna('None', inplace=True)
test.GarageQual.fillna('None', inplace=True)
test.GarageCond.fillna('None', inplace=True)
test.Alley.fillna('No_access', inplace=True)
test.MiscFeature.fillna('None', inplace=True)
test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean(), inplace=True)

```


```python
# Output too long
# train.isna().sum()
```


```python
# check if related to any other variable
train[train['MasVnrArea'].isna()]
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>234</th>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>7851</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Gilbert</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>6</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>625</td>
      <td>Unf</td>
      <td>0</td>
      <td>235</td>
      <td>860</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>860</td>
      <td>1100</td>
      <td>0</td>
      <td>1960</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>2</td>
      <td>TA</td>
      <td>BuiltIn</td>
      <td>2002.0</td>
      <td>Fin</td>
      <td>440</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>288</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>216500</td>
    </tr>
    <tr>
      <th>529</th>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>32668</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>CulDSac</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>3</td>
      <td>1957</td>
      <td>1975</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Stone</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>TA</td>
      <td>TA</td>
      <td>No</td>
      <td>Rec</td>
      <td>1219</td>
      <td>Unf</td>
      <td>0</td>
      <td>816</td>
      <td>2035</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2515</td>
      <td>0</td>
      <td>0</td>
      <td>2515</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>TA</td>
      <td>9</td>
      <td>Maj1</td>
      <td>2</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1975.0</td>
      <td>RFn</td>
      <td>484</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>200</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>3</td>
      <td>2007</td>
      <td>WD</td>
      <td>Alloca</td>
      <td>200624</td>
    </tr>
    <tr>
      <th>936</th>
      <td>20</td>
      <td>RL</td>
      <td>67.0</td>
      <td>10083</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>SawyerW</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>833</td>
      <td>Unf</td>
      <td>0</td>
      <td>343</td>
      <td>1176</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1200</td>
      <td>0</td>
      <td>0</td>
      <td>1200</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>555</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>184900</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>20</td>
      <td>RL</td>
      <td>107.0</td>
      <td>13891</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NridgHt</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>10</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Ex</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Ex</td>
      <td>Gd</td>
      <td>Gd</td>
      <td>GLQ</td>
      <td>1386</td>
      <td>Unf</td>
      <td>0</td>
      <td>690</td>
      <td>2076</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>2076</td>
      <td>0</td>
      <td>0</td>
      <td>2076</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>Ex</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2006.0</td>
      <td>Fin</td>
      <td>850</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>216</td>
      <td>229</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2006</td>
      <td>New</td>
      <td>Partial</td>
      <td>465000</td>
    </tr>
    <tr>
      <th>1278</th>
      <td>60</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9473</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>804</td>
      <td>Unf</td>
      <td>0</td>
      <td>324</td>
      <td>1128</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1128</td>
      <td>903</td>
      <td>0</td>
      <td>2031</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>2002.0</td>
      <td>RFn</td>
      <td>577</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>211</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>3</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>237000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check if related to any other variable
train[train['BsmtQual'].isna()]
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>90</td>
      <td>RL</td>
      <td>65.0</td>
      <td>6040</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>1Story</td>
      <td>4</td>
      <td>5</td>
      <td>1955</td>
      <td>1955</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>AsbShng</td>
      <td>Plywood</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>N</td>
      <td>FuseP</td>
      <td>1152</td>
      <td>0</td>
      <td>0</td>
      <td>1152</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>Fa</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>1977.336141</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>N</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>WD</td>
      <td>AdjLand</td>
      <td>82000</td>
    </tr>
    <tr>
      <th>156</th>
      <td>20</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7200</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>7</td>
      <td>1950</td>
      <td>1950</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>FuseF</td>
      <td>1040</td>
      <td>0</td>
      <td>0</td>
      <td>1040</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Detchd</td>
      <td>1950.000000</td>
      <td>Unf</td>
      <td>625</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>109500</td>
    </tr>
    <tr>
      <th>182</th>
      <td>20</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Artery</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>1957</td>
      <td>2006</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Sdng</td>
      <td>BrkFace</td>
      <td>98.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1340</td>
      <td>0</td>
      <td>0</td>
      <td>1340</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Attchd</td>
      <td>1957.000000</td>
      <td>RFn</td>
      <td>252</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>180</td>
      <td>0</td>
      <td>0</td>
      <td>MnPrv</td>
      <td>None</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>259</th>
      <td>20</td>
      <td>RM</td>
      <td>70.0</td>
      <td>12702</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>OldTown</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>1956</td>
      <td>1956</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>BrkFace</td>
      <td>BrkFace</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>PConc</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>FuseA</td>
      <td>882</td>
      <td>0</td>
      <td>0</td>
      <td>882</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>4</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Detchd</td>
      <td>1956.000000</td>
      <td>Unf</td>
      <td>308</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>97000</td>
    </tr>
    <tr>
      <th>371</th>
      <td>50</td>
      <td>RL</td>
      <td>80.0</td>
      <td>17120</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>ClearCr</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1.5Fin</td>
      <td>4</td>
      <td>4</td>
      <td>1959</td>
      <td>1959</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>WdShing</td>
      <td>Plywood</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1120</td>
      <td>468</td>
      <td>0</td>
      <td>1588</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Min2</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1991.000000</td>
      <td>Fin</td>
      <td>680</td>
      <td>TA</td>
      <td>TA</td>
      <td>N</td>
      <td>0</td>
      <td>59</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>134432</td>
    </tr>
    <tr>
      <th>520</th>
      <td>190</td>
      <td>RL</td>
      <td>60.0</td>
      <td>10800</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>OldTown</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>2fmCon</td>
      <td>2Story</td>
      <td>4</td>
      <td>7</td>
      <td>1900</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>N</td>
      <td>FuseA</td>
      <td>694</td>
      <td>600</td>
      <td>0</td>
      <td>1294</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>TA</td>
      <td>7</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>1977.336141</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>N</td>
      <td>220</td>
      <td>114</td>
      <td>210</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>8</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>106250</td>
    </tr>
    <tr>
      <th>553</th>
      <td>20</td>
      <td>RL</td>
      <td>67.0</td>
      <td>8777</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>5</td>
      <td>1949</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1126</td>
      <td>0</td>
      <td>0</td>
      <td>1126</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>5</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Detchd</td>
      <td>2002.000000</td>
      <td>Fin</td>
      <td>520</td>
      <td>TA</td>
      <td>TA</td>
      <td>N</td>
      <td>0</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>MnPrv</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>108000</td>
    </tr>
    <tr>
      <th>646</th>
      <td>20</td>
      <td>RL</td>
      <td>60.0</td>
      <td>7200</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>1950</td>
      <td>1950</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1048</td>
      <td>0</td>
      <td>0</td>
      <td>1048</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>7</td>
      <td>Min1</td>
      <td>0</td>
      <td>None</td>
      <td>Detchd</td>
      <td>1950.000000</td>
      <td>Unf</td>
      <td>420</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>98300</td>
    </tr>
    <tr>
      <th>736</th>
      <td>90</td>
      <td>RL</td>
      <td>60.0</td>
      <td>8544</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>NAmes</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>1Story</td>
      <td>3</td>
      <td>4</td>
      <td>1950</td>
      <td>1950</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Stucco</td>
      <td>Stone</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>N</td>
      <td>FuseF</td>
      <td>1040</td>
      <td>0</td>
      <td>0</td>
      <td>1040</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Detchd</td>
      <td>1949.000000</td>
      <td>Unf</td>
      <td>400</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>93500</td>
    </tr>
    <tr>
      <th>984</th>
      <td>90</td>
      <td>RL</td>
      <td>75.0</td>
      <td>10125</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Mitchel</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>1.5Fin</td>
      <td>5</td>
      <td>5</td>
      <td>1977</td>
      <td>1977</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Plywood</td>
      <td>Plywood</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1302</td>
      <td>432</td>
      <td>0</td>
      <td>1734</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Attchd</td>
      <td>1977.000000</td>
      <td>Unf</td>
      <td>539</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>8</td>
      <td>2009</td>
      <td>COD</td>
      <td>Normal</td>
      <td>126000</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>90</td>
      <td>RL</td>
      <td>75.0</td>
      <td>9825</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>Duplex</td>
      <td>1Story</td>
      <td>5</td>
      <td>5</td>
      <td>1965</td>
      <td>1965</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>AsphShn</td>
      <td>AsphShn</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>TA</td>
      <td>N</td>
      <td>SBrkr</td>
      <td>1664</td>
      <td>0</td>
      <td>0</td>
      <td>1664</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>TA</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>1977.336141</td>
      <td>None</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>100000</td>
    </tr>
    <tr>
      <th>1049</th>
      <td>20</td>
      <td>RL</td>
      <td>60.0</td>
      <td>11100</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>Edwards</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>4</td>
      <td>7</td>
      <td>1946</td>
      <td>2006</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>930</td>
      <td>0</td>
      <td>0</td>
      <td>930</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Detchd</td>
      <td>1946.000000</td>
      <td>Unf</td>
      <td>308</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>84900</td>
    </tr>
    <tr>
      <th>1321</th>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>6627</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>BrkSide</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>3</td>
      <td>6</td>
      <td>1949</td>
      <td>1950</td>
      <td>Hip</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Floor</td>
      <td>TA</td>
      <td>N</td>
      <td>SBrkr</td>
      <td>720</td>
      <td>0</td>
      <td>0</td>
      <td>720</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>TA</td>
      <td>4</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Detchd</td>
      <td>1955.000000</td>
      <td>Unf</td>
      <td>287</td>
      <td>TA</td>
      <td>Fa</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>7</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>72500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# when basemnet quality is nan, all other basement variables are nan. Indicates there is no basement
train['BsmtQual'] = train['BsmtQual'].fillna('None')
train.BsmtCond=np.where(train.BsmtQual=='None',train.BsmtCond.fillna('None'),train.BsmtCond)
train.BsmtExposure=np.where(train.BsmtQual=='None',train.BsmtExposure.fillna('None'),train.BsmtExposure)
train.BsmtFinType1=np.where(train.BsmtQual=='None',train.BsmtFinType1.fillna('None'),train.BsmtFinType1)
train.BsmtFinType2=np.where(train.BsmtQual=='None',train.BsmtFinType2.fillna('None'),train.BsmtFinType2)

# same nan handling on test set
test['BsmtQual'] = test['BsmtQual'].fillna('None')
test.BsmtCond=np.where(test.BsmtQual=='None',test.BsmtCond.fillna('None'),test.BsmtCond)
test.BsmtExposure=np.where(test.BsmtQual=='None',test.BsmtExposure.fillna('None'),test.BsmtExposure)
test.BsmtFinType1=np.where(test.BsmtQual=='None',test.BsmtFinType1.fillna('None'),test.BsmtFinType1)
test.BsmtFinType2=np.where(test.BsmtQual=='None',test.BsmtFinType2.fillna('None'),test.BsmtFinType2)

```


```python
# train.isna().sum()
```


```python
# fill residual categorical variable nans with mode
train['BsmtExposure'].fillna(train['BsmtExposure'].mode()[0], inplace=True)
train['BsmtFinType2'].fillna(train['BsmtFinType2'].mode()[0], inplace=True)
train['MasVnrType'].fillna(train['MasVnrType'].mode()[0], inplace=True)
train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)
# fill residual numerical variable nans with mean
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(), inplace=True)
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace=True)


# same on test set
test['LotFrontage'] = test.groupby('LotArea')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
test['BsmtExposure'].fillna(test['BsmtExposure'].mode()[0], inplace=True)
test['BsmtFinType2'].fillna(test['BsmtFinType2'].mode()[0], inplace=True)
test['MasVnrType'].fillna(test['MasVnrType'].mode()[0], inplace=True)
test['Electrical'].fillna(test['Electrical'].mode()[0], inplace=True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mean(), inplace=True)
test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace=True)

```


```python
train.isnull().values.any()
```




    False




```python
test.isnull().values.any()
```




    True




```python
# test.isna().sum()
```


```python
# drop rows from test set
test = test.dropna() 
test.isnull().values.any()
```




    False




```python
train.head()
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>None</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>No_access</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>



## Encoding categorical features 


```python
# create ordering for ordinal features
quality_order = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
basement_fin = {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}
functional_order = {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7}


# transform each category to its numeric order using map
train['ExterQual'] = train.ExterQual.map(quality_order)
train['ExterCond'] = train.ExterCond.map(quality_order)
train['BsmtQual'] = train.BsmtQual.map(quality_order)
train['BsmtCond'] = train.BsmtCond.map(quality_order)
train['HeatingQC'] = train.HeatingQC.map(quality_order)
train['KitchenQual'] = train.KitchenQual.map(quality_order)
train['FireplaceQu'] = train.FireplaceQu.map(quality_order)
train['GarageQual'] = train.GarageQual.map(quality_order)
train['GarageCond'] = train.GarageCond.map(quality_order)

train['BsmtFinType1'] = train.BsmtFinType1.map(basement_fin)
train['BsmtFinType2'] = train.BsmtFinType2.map(basement_fin)

train['Functional'] = train.Functional.map(functional_order)


# Do same for test data

# transform each category to its numeric order using map
test['ExterQual'] = test.ExterQual.map(quality_order)
test['ExterCond'] = test.ExterCond.map(quality_order)
test['BsmtQual'] = test.BsmtQual.map(quality_order)
test['BsmtCond'] = test.BsmtCond.map(quality_order)
test['HeatingQC'] = test.HeatingQC.map(quality_order)
test['KitchenQual'] = test.KitchenQual.map(quality_order)
test['FireplaceQu'] = test.FireplaceQu.map(quality_order)
test['GarageQual'] = test.GarageQual.map(quality_order)
test['GarageCond'] = test.GarageCond.map(quality_order)

test['BsmtFinType1'] = test.BsmtFinType1.map(basement_fin)
test['BsmtFinType2'] = test.BsmtFinType2.map(basement_fin)

test['Functional'] = test.Functional.map(functional_order)

```


```python
# dummy encode nominal categorical variables
train = pd.get_dummies(train, columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                                      'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                                      'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                                      'Exterior2nd', 'MasVnrType', 'Foundation','BsmtExposure', 
                                      'Heating', 'CentralAir', 'Electrical', 'GarageType', 'GarageFinish',
                                      'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'],
                       drop_first=True)

test = pd.get_dummies(test, columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                                      'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
                                      'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
                                      'Exterior2nd', 'MasVnrType', 'Foundation','BsmtExposure', 
                                      'Heating', 'CentralAir', 'Electrical', 'GarageType', 'GarageFinish',
                                      'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'],
                       drop_first=True)
```

## Visualize our most significant features


```python
# visualize negatively correlated features
df_corr = train.corr()['SalePrice'].sort_values(ascending=True).head(10)
df_corr.plot(kind='barh', color='navy')
plt.title('Top 10 Features Negatively Correlated to Sales Price')
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_33_0.png" alt="linearly separable data">
   




```python
# visualize positively correlated features
df_corr = train.corr()['SalePrice'].sort_values(ascending=False)[1:11]
df_corr.plot(kind='barh', color='navy')
plt.title('Top 10 Features Positively Correlated to Sales Price')
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_34_0.png" alt="linearly separable data">
    
    



```python
# split into independent and dependent variables
X = train.drop(['SalePrice'], axis=1)
y = train.SalePrice
```


```python
# Split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
```

## Remove outliers


```python
from sklearn.ensemble import IsolationForest
# Identify and remove outliers
clf = IsolationForest(random_state = 1, contamination= 'auto')
preds = clf.fit_predict(X_train)
X_train_cleaned = X_train[np.where(preds == 1, True, False)]
y_train_cleaned = y_train[np.where(preds == 1, True, False)]
print('There were {} points in the original training data. After removing outliers there are {} points.'
      .format(len(X_train),len(X_train_cleaned)))
```

    There were 969 points in the original training data. After removing outliers there are 961 points.



```python
train.shape
```




    (1385, 211)



## Scale data


```python
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler().fit(X_train_cleaned)

# transform all other data
X_train_norm = norm.transform(X_train_cleaned)
X_val_norm = norm.transform(X_val)
```

## Feature Selection


```python
fs = SelectKBest(score_func=f_regression, k=175)

# fit feature selection model on training data
fs = fs.fit(X_train_norm, y_train_cleaned)
# transform all other data
X_train_fs = fs.transform(X_train_norm)
X_val_fs = fs.transform(X_val_norm)
```


```python
# Print most important features
df_scores = pd.DataFrame(fs.scores_)
df_columns = pd.DataFrame(X_train.columns)
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Feature_Name','Score']
print(feature_scores.nlargest(10,'Score'))
```

       Feature_Name        Score
    3   OverallQual  1744.565211
    22    GrLivArea  1191.365440
    8     ExterQual   842.379956
    17  TotalBsmtSF   729.116482
    29  KitchenQual   700.834252
    19     1stFlrSF   694.994522
    35   GarageArea   684.941149
    10     BsmtQual   483.864049
    25     FullBath   407.779605
    33  FireplaceQu   394.581577



```python
top_features=feature_scores.nlargest(10,'Score')
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('whitegrid', {'axes.grid' : True})
ax = sns.barplot(x='Feature_Name', y='Score', data=top_features,
                 palette='Blues_r')
ax.set(xticklabels=['Overall Quality', 'Living Room sqft', 'Exterior Quality','Basement sqft', 'Kitchen Quality', 
                   'Main Floor sqft', 'Garage Size', 'Basement Quality', 'Full Bathrooms', 'Fireplace Quality'])
ax.set_title('Top 10 Features Driving Sales Price',fontsize= 20) 
ax.set_xlabel('')
ax.set_ylabel('F-Score')
plt.xticks(rotation=45)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
     [Text(0, 0, 'Overall Quality'),
      Text(1, 0, 'Living Room sqft'),
      Text(2, 0, 'Exterior Quality'),
      Text(3, 0, 'Basement sqft'),
      Text(4, 0, 'Kitchen Quality'),
      Text(5, 0, 'Main Floor sqft'),
      Text(6, 0, 'Garage Size'),
      Text(7, 0, 'Basement Quality'),
      Text(8, 0, 'Full Bathrooms'),
      Text(9, 0, 'Fireplace Quality')])




<img src="{{ site.url }}{{ site.baseurl }}/images/output_45_1.png" alt="linearly separable data">
    



## Linear Regression


```python
from sklearn.linear_model import LinearRegression
lin = LinearRegression().fit(X_train_fs, y_train_cleaned)
y_pred = lin.predict(X_val_fs)
```


```python
r2 = r2_score(y_pred, y_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
print("The Linear Regression model's R2 = {:.3f} and RMSE = {:.0f}".format(r2,rmse))
```

    The Linear Regression model's R2 = 0.872 and RMSE = 27514


## Ridge Regression


```python
from sklearn.linear_model import Ridge
param_grid = {'alpha': [5,6,7,8]}
ridge = Ridge()
g_search = GridSearchCV(estimator=ridge, param_grid=param_grid, verbose=0)
g_search.fit(X_train_fs, y_train_cleaned)
print(g_search.best_estimator_.alpha)
g_search.best_score_
```

    5





    0.8838328070133773




```python
# Train model with grid values
ridge = Ridge(alpha=5).fit(X_train_fs, y_train_cleaned)
# check how validation data compares
y_pred = ridge.predict(X_val_fs)
```


```python
r2 = r2_score(y_pred, y_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
print("The Ridge Regression model's R2 = {:.3f} and RMSE = {:.0f}".format(r2,rmse))
```

    The Ridge Regression model's R2 = 0.899 and RMSE = 23262


## Lasso Regression


```python
from sklearn.linear_model import Lasso
param_grid = {'alpha': [50, 55, 60, 65, 70]}
lasso = Lasso()
g_search = GridSearchCV(estimator=lasso, param_grid=param_grid)
g_search.fit(X_train_fs, y_train_cleaned)
print(g_search.best_estimator_.alpha)
g_search.best_score_
```

    55





    0.8891084881789736




```python
lasso = Lasso(alpha=55).fit(X_train_fs, y_train_cleaned)
y_pred = lasso.predict(X_val_fs)
```


```python
r2 = r2_score(y_pred, y_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
print("The Lasso Regression model's R2 = {:.3f} and RMSE = {:.0f}".format(r2,rmse))
```

    The Lasso Regression model's R2 = 0.895 and RMSE = 23892


## RandomForest Regression


```python
from sklearn.ensemble import RandomForestRegressor

param_grid ={'max_depth': [20, 35, 50],
             'max_features': ['auto', 'log2', 'sqrt'],
             'min_samples_split': [2, 3],
             'n_estimators': [200, 225, 250, 275]
            }
rfr = RandomForestRegressor(random_state = 1)
g_search = GridSearchCV(estimator = rfr, param_grid = param_grid, verbose=1, cv = 3)
g_search.fit(X_train_fs, y_train_cleaned);
print(g_search.best_params_)
g_search.best_score_
```

    Fitting 3 folds for each of 72 candidates, totalling 216 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 216 out of 216 | elapsed:  3.7min finished


    {'max_depth': 35, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 250}





    0.8815566187479771




```python
rfr = RandomForestRegressor(bootstrap=True, max_depth=35, max_features='auto',
                            min_samples_leaf=1, min_samples_split=2,
                           n_estimators=250).fit(X_train_fs, y_train_cleaned)
y_pred = rfr.predict(X_val_fs)
```


```python
r2 = r2_score(y_pred, y_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
print("The RandomForest Regression model's R2 = {:.3f} and RMSE = {:.0f}".format(r2,rmse))
```

    The RandomForest Regression model's R2 = 0.868 and RMSE = 25053


## Decision Tree Regression


```python
from sklearn.tree import DecisionTreeRegressor

param_grid = {'max_depth': [5,10,15],
              'min_samples_split': [2,3],
             'min_samples_leaf': [2,3,4,5],
             'max_features': ['auto', 'sqrt', 'log2', None],
             'max_leaf_nodes': [45, 55, 60, 65]
             }
tree = DecisionTreeRegressor(random_state=1)
g_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=3, verbose=1)
g_search.fit(X_train_fs, y_train_cleaned);
print(g_search.best_params_)
g_search.best_score_
```

    Fitting 3 folds for each of 384 candidates, totalling 1152 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    {'max_depth': 10, 'max_features': 'auto', 'max_leaf_nodes': 60, 'min_samples_leaf': 4, 'min_samples_split': 2}


    [Parallel(n_jobs=1)]: Done 1152 out of 1152 | elapsed:    7.1s finished





    0.7502263971260735




```python
tree = DecisionTreeRegressor(max_depth=10, max_features='auto', max_leaf_nodes=60, min_samples_leaf=4,
                             min_samples_split=2).fit(X_train_fs, y_train_cleaned)
y_pred = tree.predict(X_val_fs)
```


```python
r2 = r2_score(y_pred, y_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
print("The Decision Tree Regression model's R2 = {:.3f} and RMSE = {:.0f}".format(r2,rmse))
```

    The Decision Tree Regression model's R2 = 0.768 and RMSE = 33264


## Support Vector Regression


```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
param_grid = {'kernel': ('linear', 'rbf','poly','sigmoid'), 'C':[1000, 5000, 10000],
              'gamma': [1e-12, 1e-11, 1e-10],'epsilon':[0.5, 1]
             }
svr = SVR()
g_search = GridSearchCV(svr, param_grid, cv=3, verbose=1).fit(X_train_fs,y_train_cleaned)
print(g_search.best_params_)
g_search.best_score_
```

    Fitting 3 folds for each of 72 candidates, totalling 216 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 216 out of 216 | elapsed:   23.7s finished


    {'C': 10000, 'epsilon': 1, 'gamma': 1e-12, 'kernel': 'linear'}





    0.8644226746456107




```python
regressor = SVR(kernel = 'linear', C=10000).fit(X_train_fs, y_train_cleaned)
y_pred = regressor.predict(X_val_fs)
```


```python
r2 = r2_score(y_pred, y_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
print("The Support Vector Regression model's R2 = {:.3f} and RMSE = {:.0f}".format(r2,rmse))
```

    The Support Vector Regression model's R2 = 0.863 and RMSE = 24813


## XGBoost Regression


```python
from xgboost import XGBRegressor
param_grid = {'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [4, 5],
        'subsample': [0.5, 0.6],
        'colsample_bytree': [0.6, 0.7],
        'n_estimators' : [1000, 1500, 2000]
    }
xgb = XGBRegressor()
g_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, verbose=1).fit(X_train_fs, y_train_cleaned)
print(g_search.best_params_)
g_search.best_score_
```

    Fitting 3 folds for each of 144 candidates, totalling 432 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 432 out of 432 | elapsed: 17.2min finished


    {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 4, 'n_estimators': 2000, 'subsample': 0.6}





    0.9171461076074751




```python
xgb = XGBRegressor(learning_rate =0.01, max_depth=5, min_child_weight=4, subsample=0.6,
                  colsample_bytree= 0.7, n_estimators=2000).fit(X_train_fs, y_train_cleaned)
y_pred = xgb.predict(X_val_fs)
```


```python
r2 = r2_score(y_pred, y_val)
mse = mean_squared_error(y_pred, y_val)
rmse = np.sqrt(mse)
print("The XGBoost Regression model's R2 = {:.3f} and RMSE = {:.0f}".format(r2,rmse))
```

    The XGBoost Regression model's R2 = 0.924 and RMSE = 19867


## Visualizing Model Performance


```python
from yellowbrick.regressor import PredictionError, ResidualsPlot
```


```python
model1 = xgb
visualizer = PredictionError(model1, line_color='black')
visualizer.fit(X_train_fs, y_train_cleaned)  
visualizer.score(X_val_fs, y_val)  
visualizer.poof()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_75_0.png" alt="linearly separable data">
    
    





    <AxesSubplot:title={'center':'Prediction Error for XGBRegressor'}, xlabel='$y$', ylabel='$\\hat{y}$'>




```python
visualizer = ResidualsPlot(model1, test_color='pink', test_alpha=0.7, train_color='#357EC7',
                          train_alpha=0.6, line_color='grey')
visualizer.fit(X_train_fs, y_train_cleaned)  
visualizer.score(X_val_fs, y_val)    
visualizer.poof()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/output_76_0.png" alt="linearly separable data">
    
    





    <AxesSubplot:title={'center':'Residuals for XGBRegressor Model'}, xlabel='Predicted Value', ylabel='Residuals'>




```python

```


```python

```
