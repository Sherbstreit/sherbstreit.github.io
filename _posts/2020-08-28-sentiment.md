---
title: "Sentiment Analysis on New York Times Articles"
date: 2020-08-28
tags: [data wrangling, data science, Sentiment analysis]
header:
  image:
excerpt: "Extracting sentiment scores from New York Times Articles"
mathjax: "true"
---


## Sentiment Analysis on New York Times Articles


```python
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
```


```python
# load df and view
df = pd.read_csv("DailyComments.csv")
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
      <th>Day of Week</th>
      <th>comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Monday</td>
      <td>Hello, how are you?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tuesday</td>
      <td>Today is a good day!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wednesday</td>
      <td>It's my birthday so it's a really special day!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thursday</td>
      <td>Today is neither a good day or a bad day!</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Friday</td>
      <td>I'm having a bad day.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Saturday</td>
      <td>There' s nothing special happening today.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sunday</td>
      <td>Today is a SUPER good day!</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create variable for sentiment analyser
run_sentiment = SentimentIntensityAnalyzer()
```


```python
# Apply Vader analyzer to comments
df['compound'] = [run_sentiment.polarity_scores(v)['compound'] for v in df['comments']]
df['negative'] = [run_sentiment.polarity_scores(v)['neg'] for v in df['comments']]
df['neutral'] = [run_sentiment.polarity_scores(v)['neu'] for v in df['comments']]
df['positive'] = [run_sentiment.polarity_scores(v)['pos'] for v in df['comments']]
```


```python
# View sentiment analysis results
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
      <th>Day of Week</th>
      <th>comments</th>
      <th>compound</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Monday</td>
      <td>Hello, how are you?</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tuesday</td>
      <td>Today is a good day!</td>
      <td>0.4926</td>
      <td>0.000</td>
      <td>0.556</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wednesday</td>
      <td>It's my birthday so it's a really special day!</td>
      <td>0.5081</td>
      <td>0.000</td>
      <td>0.709</td>
      <td>0.291</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Thursday</td>
      <td>Today is neither a good day or a bad day!</td>
      <td>-0.7350</td>
      <td>0.437</td>
      <td>0.563</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Friday</td>
      <td>I'm having a bad day.</td>
      <td>-0.5423</td>
      <td>0.467</td>
      <td>0.533</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Saturday</td>
      <td>There' s nothing special happening today.</td>
      <td>-0.3089</td>
      <td>0.311</td>
      <td>0.689</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sunday</td>
      <td>Today is a SUPER good day!</td>
      <td>0.8327</td>
      <td>0.000</td>
      <td>0.338</td>
      <td>0.662</td>
    </tr>
  </tbody>
</table>
</div>




```python
# extra credit
# load second df
nyt_df = pd.read_csv("ArticlesApril2017.csv")
nyt_df.head()
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
      <th>abstract</th>
      <th>articleID</th>
      <th>articleWordCount</th>
      <th>byline</th>
      <th>documentType</th>
      <th>headline</th>
      <th>keywords</th>
      <th>multimedia</th>
      <th>newDesk</th>
      <th>printPage</th>
      <th>pubDate</th>
      <th>sectionName</th>
      <th>snippet</th>
      <th>source</th>
      <th>typeOfMaterial</th>
      <th>webURL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>58def1347c459f24986d7c80</td>
      <td>716</td>
      <td>By STEPHEN HILTNER and SUSAN LEHMAN</td>
      <td>article</td>
      <td>Finding an Expansive View  of a Forgotten Peop...</td>
      <td>['Photography', 'New York Times', 'Niger', 'Fe...</td>
      <td>3</td>
      <td>Insider</td>
      <td>2</td>
      <td>2017-04-01 00:15:41</td>
      <td>Unknown</td>
      <td>One of the largest photo displays in Times his...</td>
      <td>The New York Times</td>
      <td>News</td>
      <td>https://www.nytimes.com/2017/03/31/insider/nig...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>58def3237c459f24986d7c84</td>
      <td>823</td>
      <td>By GAIL COLLINS</td>
      <td>article</td>
      <td>And Now,  the Dreaded Trump Curse</td>
      <td>['United States Politics and Government', 'Tru...</td>
      <td>3</td>
      <td>OpEd</td>
      <td>23</td>
      <td>2017-04-01 00:23:58</td>
      <td>Unknown</td>
      <td>Meet the gang from under the bus.</td>
      <td>The New York Times</td>
      <td>Op-Ed</td>
      <td>https://www.nytimes.com/2017/03/31/opinion/and...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>58def9f57c459f24986d7c90</td>
      <td>575</td>
      <td>By THE EDITORIAL BOARD</td>
      <td>article</td>
      <td>Venezuela’s Descent Into Dictatorship</td>
      <td>['Venezuela', 'Politics and Government', 'Madu...</td>
      <td>3</td>
      <td>Editorial</td>
      <td>22</td>
      <td>2017-04-01 00:53:06</td>
      <td>Unknown</td>
      <td>A court ruling annulling the legislature’s aut...</td>
      <td>The New York Times</td>
      <td>Editorial</td>
      <td>https://www.nytimes.com/2017/03/31/opinion/ven...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>58defd317c459f24986d7c95</td>
      <td>1374</td>
      <td>By MICHAEL POWELL</td>
      <td>article</td>
      <td>Stain Permeates Basketball Blue Blood</td>
      <td>['Basketball (College)', 'University of North ...</td>
      <td>3</td>
      <td>Sports</td>
      <td>1</td>
      <td>2017-04-01 01:06:52</td>
      <td>College Basketball</td>
      <td>For two decades, until 2013, North Carolina en...</td>
      <td>The New York Times</td>
      <td>News</td>
      <td>https://www.nytimes.com/2017/03/31/sports/ncaa...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>58df09b77c459f24986d7ca7</td>
      <td>708</td>
      <td>By DEB AMLEN</td>
      <td>article</td>
      <td>Taking Things for Granted</td>
      <td>['Crossword Puzzles']</td>
      <td>3</td>
      <td>Games</td>
      <td>0</td>
      <td>2017-04-01 02:00:14</td>
      <td>Unknown</td>
      <td>In which Howard Barkin and Will Shortz teach u...</td>
      <td>The New York Times</td>
      <td>News</td>
      <td>https://www.nytimes.com/2017/03/31/crosswords/...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# only keep headline and media type from df
sub_df = nyt_df.loc[:,['headline', 'typeOfMaterial']]
sub_df
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
      <th>headline</th>
      <th>typeOfMaterial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finding an Expansive View  of a Forgotten Peop...</td>
      <td>News</td>
    </tr>
    <tr>
      <th>1</th>
      <td>And Now,  the Dreaded Trump Curse</td>
      <td>Op-Ed</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venezuela’s Descent Into Dictatorship</td>
      <td>Editorial</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stain Permeates Basketball Blue Blood</td>
      <td>News</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Taking Things for Granted</td>
      <td>News</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>881</th>
      <td>Reporting on Gays Who ‘Don’t Exist’</td>
      <td>News</td>
    </tr>
    <tr>
      <th>882</th>
      <td>The Fights That Could Lead to a Government Shu...</td>
      <td>News</td>
    </tr>
    <tr>
      <th>883</th>
      <td>‘The Leftovers’ Season 3, Episode 2: Swedish P...</td>
      <td>Review</td>
    </tr>
    <tr>
      <th>884</th>
      <td>Thinking Out Loud, But Why?</td>
      <td>Review</td>
    </tr>
    <tr>
      <th>885</th>
      <td>Some Sugar. Could Use More Spice.</td>
      <td>Review</td>
    </tr>
  </tbody>
</table>
<p>886 rows × 2 columns</p>
</div>




```python
# Apply Vader analyzer to headline
sub_df['compound'] = [run_sentiment.polarity_scores(v)['compound'] for v in sub_df['headline']]
sub_df['negative'] = [run_sentiment.polarity_scores(v)['neg'] for v in sub_df['headline']]
sub_df['neutral'] = [run_sentiment.polarity_scores(v)['neu'] for v in sub_df['headline']]
sub_df['positive'] = [run_sentiment.polarity_scores(v)['pos'] for v in sub_df['headline']]
```


```python
# View sentiment analysis results
sub_df.head(20)
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
      <th>headline</th>
      <th>typeOfMaterial</th>
      <th>compound</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Finding an Expansive View  of a Forgotten Peop...</td>
      <td>News</td>
      <td>-0.2263</td>
      <td>0.174</td>
      <td>0.826</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>And Now,  the Dreaded Trump Curse</td>
      <td>Op-Ed</td>
      <td>-0.8020</td>
      <td>0.643</td>
      <td>0.357</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Venezuela’s Descent Into Dictatorship</td>
      <td>Editorial</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stain Permeates Basketball Blue Blood</td>
      <td>News</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Taking Things for Granted</td>
      <td>News</td>
      <td>0.2500</td>
      <td>0.000</td>
      <td>0.600</td>
      <td>0.400</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The Caged Beast Awakens</td>
      <td>Review</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>An Ever-Unfolding Story</td>
      <td>News</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>O’Reilly Thrives as Settlements Add Up</td>
      <td>News</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Mouse Infestation</td>
      <td>News</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divide in G.O.P. Now Threatens Trump Tax Plan</td>
      <td>News</td>
      <td>-0.3818</td>
      <td>0.271</td>
      <td>0.729</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Variety Puzzle: Acrostic</td>
      <td>News</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>They Can Hit a Ball 400 Feet. But Play Catch? ...</td>
      <td>News</td>
      <td>0.2960</td>
      <td>0.127</td>
      <td>0.667</td>
      <td>0.207</td>
    </tr>
    <tr>
      <th>12</th>
      <td>In Trump Country, Shock at Trump Budget Cuts</td>
      <td>Op-Ed</td>
      <td>-0.5859</td>
      <td>0.444</td>
      <td>0.556</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Why Is This Hate Different From All Other Hate?</td>
      <td>Op-Ed</td>
      <td>-0.8432</td>
      <td>0.536</td>
      <td>0.464</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Pick Your Favorite Ethical Offender</td>
      <td>Editorial</td>
      <td>0.5859</td>
      <td>0.231</td>
      <td>0.185</td>
      <td>0.583</td>
    </tr>
    <tr>
      <th>15</th>
      <td>My Son’s Growing Black Pride</td>
      <td>Op-Ed</td>
      <td>0.4767</td>
      <td>0.000</td>
      <td>0.423</td>
      <td>0.577</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Jerks and the Start-Ups They Ruin</td>
      <td>Op-Ed</td>
      <td>-0.7096</td>
      <td>0.596</td>
      <td>0.404</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Trump  Needs  a Brain</td>
      <td>Op-Ed</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Manhood in the Age of Trump</td>
      <td>Op-Ed</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>The Value of a Black College</td>
      <td>Op-Ed</td>
      <td>0.3400</td>
      <td>0.000</td>
      <td>0.676</td>
      <td>0.324</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
