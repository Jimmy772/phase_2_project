# Additional resources
[Presention Slides](https://docs.google.com/presentation/d/1p0_cDG_QNX4qFHGw9xUdtcI05w9TCC56n6j_sxu61-Y/edit?usp=sharing)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

sns.set()
```

# Problem Statement
The goal of this data science project is to help a real estate agency provide guidance to homeowners who want to buy or sell homes. One of the challenges that homeowners face is deciding whether to invest in home renovations and how much they can expect to gain from them. The project aims to address this challenge by developing a model that can estimate the value of a home based on its features and suggest the most profitable renovations for each home.
![House For Sale](visuals/house_for_sale.jpg)

# Data Understanding
In this section of the data science project, we will explore and gain a comprehensive understanding of the dataset. Understanding the data is crucial before proceeding with any analysis or modeling tasks. Let's examine the columns and their descriptions to familiarize ourselves with the dataset:

1. id: Unique identifier for each house.

2. date: Date when the house was sold.

3. price: Sale price of the house (prediction target variable).

4. bedrooms: Number of bedrooms in the house.

5. bathrooms: Number of bathrooms in the house.

6. sqft_living: Square footage of living space in the home.

7. sqft_lot: Square footage of the lot.

8. floors: Number of floors (levels) in the house.

9. waterfront: Indicates whether the house is located on a waterfront. 

10. view: Quality of the view from the house.

11. condition: Overall condition of the house, related to the maintenance of the property. .

12. grade: Overall grade of the house, related to the construction and design of the property. 

13. sqft_above: Square footage of the house apart from the basement.

14. sqft_basement: Square footage of the basement in the house.

15. yr_built: Year when the house was built.

16. yr_renovated: Year when the house was last renovated.

17. zipcode: ZIP Code used by the United States Postal Service.

18. lat: Latitude coordinate of the house.

19. long: Longitude coordinate of the house.

20. sqft_living15: The square footage of interior housing living space for the nearest 15 neighbors.

21. sqft_lot15: The square footage of the land lots of the nearest 15 neighbors.

By understanding the dataset's columns and their descriptions, we can make initial observations and formulate questions for further analysis. This step helps us gain insights into the data's nature, identify potential relationships, and plan subsequent steps for data preparation, exploration, and modeling.


```python
df = pd.read_csv("data/raw/kc_house_data.csv")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  object 
     9   view           21534 non-null  object 
     10  condition      21597 non-null  object 
     11  grade          21597 non-null  object 
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(6), int64(9), object(6)
    memory usage: 3.5+ MB
    

# Data Cleaning

## Handling missing values
![Missing Values](visuals/missing_values.jpg)
### Columns with missing values:
- waterfront - Change null to "not provided"
- view - Change null to "not provided"<br><br>
The above approach ensures that the missingness is preserved and can be taken into account during analysis.
<br><br>
- yr_renovated - Assuming the null values as well as values that are 0 mean the house has not been renovated
- zipcode
- sqft_basement - Assuming that missing values ('?') means no basement. Replce '?' with 0


```python
df.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>7129300520</td>
      <td>6414100192</td>
      <td>5631500400</td>
      <td>2487200875</td>
      <td>1954400510</td>
    </tr>
    <tr>
      <th>date</th>
      <td>10/13/2014</td>
      <td>12/9/2014</td>
      <td>2/25/2015</td>
      <td>12/9/2014</td>
      <td>2/18/2015</td>
    </tr>
    <tr>
      <th>price</th>
      <td>221900.0</td>
      <td>538000.0</td>
      <td>180000.0</td>
      <td>604000.0</td>
      <td>510000.0</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>1.0</td>
      <td>2.25</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>sqft_living</th>
      <td>1180</td>
      <td>2570</td>
      <td>770</td>
      <td>1960</td>
      <td>1680</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>5650</td>
      <td>7242</td>
      <td>10000</td>
      <td>5000</td>
      <td>8080</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>NaN</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>view</th>
      <td>NONE</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>NONE</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>Average</td>
      <td>Average</td>
      <td>Average</td>
      <td>Very Good</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>7 Average</td>
      <td>7 Average</td>
      <td>6 Low Average</td>
      <td>7 Average</td>
      <td>8 Good</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>1180</td>
      <td>2170</td>
      <td>770</td>
      <td>1050</td>
      <td>1680</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>0.0</td>
      <td>400.0</td>
      <td>0.0</td>
      <td>910.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>yr_built</th>
      <td>1955</td>
      <td>1951</td>
      <td>1933</td>
      <td>1965</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>yr_renovated</th>
      <td>0.0</td>
      <td>1991.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>zipcode</th>
      <td>98178</td>
      <td>98125</td>
      <td>98028</td>
      <td>98136</td>
      <td>98074</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>47.5112</td>
      <td>47.721</td>
      <td>47.7379</td>
      <td>47.5208</td>
      <td>47.6168</td>
    </tr>
    <tr>
      <th>long</th>
      <td>-122.257</td>
      <td>-122.319</td>
      <td>-122.233</td>
      <td>-122.393</td>
      <td>-122.045</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>1340</td>
      <td>1690</td>
      <td>2720</td>
      <td>1360</td>
      <td>1800</td>
    </tr>
    <tr>
      <th>sqft_lot15</th>
      <td>5650</td>
      <td>7639</td>
      <td>8062</td>
      <td>5000</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["waterfront"].value_counts(dropna=False)
```




    NO     19075
    NaN     2376
    YES      146
    Name: waterfront, dtype: int64




```python
df["view"].value_counts(dropna=False)
```




    NONE         19422
    AVERAGE        957
    GOOD           508
    FAIR           330
    EXCELLENT      317
    NaN             63
    Name: view, dtype: int64




```python
columns = ["view", "waterfront"]
for column in columns:
    df[column].fillna("not provided", inplace=True)
```


```python
df[columns].info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 2 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   view        21597 non-null  object
     1   waterfront  21597 non-null  object
    dtypes: object(2)
    memory usage: 337.6+ KB
    


```python
df["yr_renovated"].replace(0, np.nan, inplace=True)  # replace 0 with NaN
df["yr_renovated"].fillna("not renovated", inplace=True)  # fillna with "not renovated"
```


```python
df["sqft_basement"].replace("?", 0, inplace=True)  # replace '?' with 0
df = df.astype({"sqft_basement": float})  # change from object to float
df["sqft_basement"].dtype  # confirm changes
```




    dtype('float64')



# EDA
![EDA](visuals/eda.jfif)

## Univariante analysis
### Findings:
- A majority of our numeric variables as skewed to the right
- A mjority of our numeric variables have ouliers
    - We'll need to address this before adding them to our model
- A majority of our categorical variables are imbalanced.


```python
df.describe()
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
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.159700e+04</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.580474e+09</td>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>1788.596842</td>
      <td>285.716581</td>
      <td>1970.999676</td>
      <td>98077.951845</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.876736e+09</td>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>827.759761</td>
      <td>439.819830</td>
      <td>29.375234</td>
      <td>53.513072</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>370.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.123049e+09</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>1190.000000</td>
      <td>0.000000</td>
      <td>1951.000000</td>
      <td>98033.000000</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904930e+09</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>1975.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.308900e+09</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>2210.000000</td>
      <td>550.000000</td>
      <td>1997.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Visulizing distribuition of numerc variables


```python
# list for the continous numeric variables
cont_cols = [
    "sqft_basement",
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_living15",
    "sqft_lot15",
]

# list for the discrete numeric variables
dic_cols = ["bedrooms", "bathrooms", "floors"]
```


```python
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

axs = [item for sublist in axs for item in sublist]

for idx, cont_col in enumerate(cont_cols + dic_cols):
    sns.histplot(data=df, x=cont_col, ax=axs[idx])
    axs[idx].axes.get_yaxis().set_visible(False)  # turning of y axis
```


    
![png](index_files/index_19_0.png)
    



```python
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

axs = [item for sublist in axs for item in sublist]

for idx, cont_col in enumerate(cont_cols + dic_cols):
    sns.boxplot(x=df[cont_col], ax=axs[idx])
    axs[idx].axes.get_yaxis().set_visible(False)  # turning of y axis
```


    
![png](index_files/index_20_0.png)
    


### Visulizing distribution of categorical features


```python
# list of categorical variables
cat_cols = ["grade", "view", "condition", "waterfront"]
```


```python
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

axs = [item for sublist in axs for item in sublist]

for idx, cat_col in enumerate(cat_cols):
    # define data
    data = df[cat_col].value_counts().values
    labels = df[cat_col].value_counts().index

    # create pie chart
    axs[idx].pie(data, autopct="%.0f%%")
    # set title
    axs[idx].set_title(f"{cat_col.title()} Distribution")
    # create legend
    axs[idx].legend(labels)
```


    
![png](index_files/index_23_0.png)
    


## Bivariate analysis
In this section well be looking at how our independent variables are related to the dependent variable (price)

### Findings:
- All numeric variables have a postive correlation with price
- Most correlated are sqft_living, sqft_above, sqrt_living15, and bathrooms
- Bedrooms and floors have a medium positive correlation.
- Those 6 variables will be useful when creating our model.
- Since the p-value is less than 0.05 for all the One-Way Anova tests conducted on the categorical feature, we can reject the null hypothesis. This implies that we have sufficient proff to say that their exists a difference in price amoung the different categories in each categorical variable.
- Location also seems to have an effect on price, with most of the expensive houses are clustered together.

### Numeric variables


```python
# list of numeric columns
num_cols = [
    "price",
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "sqft_living15",
    "sqft_lot15",
    "bedrooms",
    "bathrooms",
    "floors",
]
corr = df[num_cols].corr()
corr
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
      <th>price</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>floors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>1.000000</td>
      <td>0.701917</td>
      <td>0.089876</td>
      <td>0.605368</td>
      <td>0.321108</td>
      <td>0.585241</td>
      <td>0.082845</td>
      <td>0.308787</td>
      <td>0.525906</td>
      <td>0.256804</td>
    </tr>
    <tr>
      <th>sqft_living</th>
      <td>0.701917</td>
      <td>1.000000</td>
      <td>0.173453</td>
      <td>0.876448</td>
      <td>0.428660</td>
      <td>0.756402</td>
      <td>0.184342</td>
      <td>0.578212</td>
      <td>0.755758</td>
      <td>0.353953</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>0.089876</td>
      <td>0.173453</td>
      <td>1.000000</td>
      <td>0.184139</td>
      <td>0.015031</td>
      <td>0.144763</td>
      <td>0.718204</td>
      <td>0.032471</td>
      <td>0.088373</td>
      <td>-0.004814</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>0.605368</td>
      <td>0.876448</td>
      <td>0.184139</td>
      <td>1.000000</td>
      <td>-0.051175</td>
      <td>0.731767</td>
      <td>0.195077</td>
      <td>0.479386</td>
      <td>0.686668</td>
      <td>0.523989</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>0.321108</td>
      <td>0.428660</td>
      <td>0.015031</td>
      <td>-0.051175</td>
      <td>1.000000</td>
      <td>0.199288</td>
      <td>0.015885</td>
      <td>0.297229</td>
      <td>0.278485</td>
      <td>-0.241866</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>0.585241</td>
      <td>0.756402</td>
      <td>0.144763</td>
      <td>0.731767</td>
      <td>0.199288</td>
      <td>1.000000</td>
      <td>0.183515</td>
      <td>0.393406</td>
      <td>0.569884</td>
      <td>0.280102</td>
    </tr>
    <tr>
      <th>sqft_lot15</th>
      <td>0.082845</td>
      <td>0.184342</td>
      <td>0.718204</td>
      <td>0.195077</td>
      <td>0.015885</td>
      <td>0.183515</td>
      <td>1.000000</td>
      <td>0.030690</td>
      <td>0.088303</td>
      <td>-0.010722</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>0.308787</td>
      <td>0.578212</td>
      <td>0.032471</td>
      <td>0.479386</td>
      <td>0.297229</td>
      <td>0.393406</td>
      <td>0.030690</td>
      <td>1.000000</td>
      <td>0.514508</td>
      <td>0.177944</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.525906</td>
      <td>0.755758</td>
      <td>0.088373</td>
      <td>0.686668</td>
      <td>0.278485</td>
      <td>0.569884</td>
      <td>0.088303</td>
      <td>0.514508</td>
      <td>1.000000</td>
      <td>0.502582</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>0.256804</td>
      <td>0.353953</td>
      <td>-0.004814</td>
      <td>0.523989</td>
      <td>-0.241866</td>
      <td>0.280102</td>
      <td>-0.010722</td>
      <td>0.177944</td>
      <td>0.502582</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(corr, cmap="Blues");
```


    
![png](index_files/index_27_0.png)
    


#### Continous numeric variables
Here we look at how our continous numeric variables are related to the price


```python
# list of continous variables with a medium to high correlation to
high_corr_cols = ["sqft_living", "sqft_above", "sqft_living15", "bathrooms"]
```


```python
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 15))

axs = [item for sublist in axs for item in sublist]

for idx, _ in enumerate(high_corr_cols):
    sns.scatterplot(x=_, y="price", data=df, ax=axs[idx])
    axs[idx].set_title(f"{_} against price")
```


    
![png](index_files/index_30_0.png)
    


### Discrete numeric variables


```python
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 8))

for idx, _ in enumerate(dic_cols):
    sns.barplot(x="price", y=_, data=df, ax=axs[idx], orient="h")
    axs[idx].set_title(f"price against {_}")
```


    
![png](index_files/index_32_0.png)
    


### Geographic numeric features


```python
def plot_coordinates_on_map(dataframe, lat_column, lon_column, value_column):
    # Use Plotly Express to create a scatter mapbox plot
    fig = px.scatter_mapbox(
        dataframe,
        lat=lat_column,
        lon=lon_column,
        zoom=10,
        hover_data=[value_column],
        color=value_column,
        color_continuous_scale="Viridis",
    )

    # Update the map layout
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Display the map
    fig.show()
```


```python
# Call the function to plot coordinates on a map
plot_coordinates_on_map(df, "lat", "long", "price")
```



### Categorical variables
We will use column charts to visulize how the mean house price changes by categorical varible.

We'll also perform a 1-way ANOVA to check if the mean house price is different by the categorical variable.


```python
def perform_one_way_anova(dataframe, categorical_column, label_column):
    # Get unique categories from the categorical column
    categories = dataframe[categorical_column].unique()

    # Create a dictionary to store the category and corresponding label values
    category_data = {}

    # Populate the dictionary with category and label values
    for category in categories:
        category_data[category] = dataframe[dataframe[categorical_column] == category][
            label_column
        ]

    # Perform one-way ANOVA test
    statistic, p_value = f_oneway(*category_data.values())

    # Round the statistic and p-value to 4 decimal places
    statistic = round(statistic, 4)
    p_value = round(p_value, 4)

    # Return the rounded statistic and p-value
    return statistic, p_value
```


```python
def visualize_mean_by_category(dataframe, categorical_column, label_column, axis):
    # Calculate mean price by the categorical variable
    mean_prices = dataframe.groupby(categorical_column)[label_column].mean()

    # Perform one-way ANOVA test
    statistic, p_value = perform_one_way_anova(
        dataframe, categorical_column, label_column
    )

    # Create a bar plot on the specified axis
    mean_prices.plot(kind="bar", ax=axis)

    # Set the labels and title
    axis.set_xlabel(categorical_column)
    axis.set_ylabel("Mean " + label_column.title())
    axis.set_title(f"Mean {label_column.title()} by {categorical_column.title()}")

    # Display the statistic and p-value
    axis.text(
        0.05,
        0.95,
        f"Statistic: {statistic}\np-value: {p_value}",
        transform=axis.transAxes,
        fontsize=10,
        verticalalignment="top",
    )
```


```python
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 15))

axs = [item for sublist in axs for item in sublist]

for idx, _ in enumerate(cat_cols):
    visualize_mean_by_category(df, _, "price", axs[idx])
```


    
![png](index_files/index_39_0.png)
    


# Data Processing
![preprocessing](visuals/preprocessing.jpeg)
Objectives:
- Encode categorical features
    - we'll be using an ordinal encoder on three categorical features (grade, view, condition) because an ordered relationship exists in all of them
    - we'll use a one hot encoder for the waterfront feature because its values have no ordinal relationship



```python
def encode_columns(dataframe, column_name, ordinal_order):
    # Create an instance of the OrdinalEncoder with the specified order
    encoder = OrdinalEncoder(categories=[ordinal_order])

    # Reshape the column values to a 2D array
    column_values = dataframe[column_name].values.reshape(-1, 1)

    # Encode the specified column
    encoded_col = encoder.fit_transform(column_values)

    # Update the dataframe with the encoded values
    dataframe[column_name] = encoded_col

    # Return the updated dataframe
    return dataframe
```


```python
orders_dict = {  # a dict to hold the ordinal order for all cat features
    "grade": [
        "3 Poor",
        "4 Low",
        "5 Fair",
        "6 Low Average",
        "7 Average",
        "8 Good",
        "9 Better",
        "10 Very Good",
        "11 Excellent",
        "12 Luxury",
        "13 Mansion",
    ],
    "view": ["not provided", "NONE", "FAIR", "AVERAGE", "GOOD", "EXCELLENT"],
    "condition": ["Poor", "Fair", "Average", "Good", "Very Good"],
}
```


```python
# create a df to hold encoded data
df_encoded = df.copy()
```


```python
for column, order in orders_dict.items():
    df_encoded = encode_columns(df_encoded, column, order)
    print(f"{column} encoded")
```

    grade encoded
    view encoded
    condition encoded
    


```python
def one_hot_encode(dataframe, column_name):
    # Create an instance of the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Reshape the column values to a 2D array
    values = dataframe[column_name].values.reshape(-1, 1)

    # Perform one-hot encoding
    encoded = encoder.fit_transform(values)

    # Create column names for the one-hot encoded features
    categories = encoder.categories_[0]
    column_names = [f"{column_name}_{category}" for category in categories]

    # Create a new dataframe with the one-hot encoded features
    dataframe_encoded = pd.DataFrame(encoded, columns=column_names)

    # Concatenate the original dataframe with the one-hot encoded dataframe
    dataframe_encoded = pd.concat([dataframe, dataframe_encoded], axis=1)

    # Drop original column
    dataframe_encoded = dataframe_encoded.drop(column_name, axis=1)

    # Return the updated dataframe
    return dataframe_encoded
```


```python
# apply one hot encoding to waterfront column
df_encoded = one_hot_encode(df_encoded, "waterfront")

# save file
df_encoded.to_csv(path_or_buf="data/processed/encoded_data.csv", index=False)
```

# Modeling
![modeling](visuals/modeling.jpeg)
We'll be creating a linear regression model.

Every model has makes its own assumtions about the data. The following are the assumtions made my a linear regression model:
- Linear relationship
- Multivariate normality
- No or little multicollinearity
- Homescedasticity

## Checking for multicollinearity
We'll first start by checking for multicollinearity to help in selcting which features to include in our model


```python
data = pd.read_csv("data/processed/encoded_data.csv")
```


```python
data["zipcode"].nunique()
```




    70




```python
data.head().T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>7129300520</td>
      <td>6414100192</td>
      <td>5631500400</td>
      <td>2487200875</td>
      <td>1954400510</td>
    </tr>
    <tr>
      <th>date</th>
      <td>10/13/2014</td>
      <td>12/9/2014</td>
      <td>2/25/2015</td>
      <td>12/9/2014</td>
      <td>2/18/2015</td>
    </tr>
    <tr>
      <th>price</th>
      <td>221900.0</td>
      <td>538000.0</td>
      <td>180000.0</td>
      <td>604000.0</td>
      <td>510000.0</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>1.0</td>
      <td>2.25</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>sqft_living</th>
      <td>1180</td>
      <td>2570</td>
      <td>770</td>
      <td>1960</td>
      <td>1680</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>5650</td>
      <td>7242</td>
      <td>10000</td>
      <td>5000</td>
      <td>8080</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>view</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>1180</td>
      <td>2170</td>
      <td>770</td>
      <td>1050</td>
      <td>1680</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>0.0</td>
      <td>400.0</td>
      <td>0.0</td>
      <td>910.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>yr_built</th>
      <td>1955</td>
      <td>1951</td>
      <td>1933</td>
      <td>1965</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>yr_renovated</th>
      <td>not renovated</td>
      <td>1991.0</td>
      <td>not renovated</td>
      <td>not renovated</td>
      <td>not renovated</td>
    </tr>
    <tr>
      <th>zipcode</th>
      <td>98178</td>
      <td>98125</td>
      <td>98028</td>
      <td>98136</td>
      <td>98074</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>47.5112</td>
      <td>47.721</td>
      <td>47.7379</td>
      <td>47.5208</td>
      <td>47.6168</td>
    </tr>
    <tr>
      <th>long</th>
      <td>-122.257</td>
      <td>-122.319</td>
      <td>-122.233</td>
      <td>-122.393</td>
      <td>-122.045</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>1340</td>
      <td>1690</td>
      <td>2720</td>
      <td>1360</td>
      <td>1800</td>
    </tr>
    <tr>
      <th>sqft_lot15</th>
      <td>5650</td>
      <td>7639</td>
      <td>8062</td>
      <td>5000</td>
      <td>7503</td>
    </tr>
    <tr>
      <th>waterfront_NO</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>waterfront_YES</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>waterfront_not provided</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# secpeciy columns to exclude
exclude_cols = [
    "price",
    "id",
    "date",
    "yr_renovated",
    "yr_built",
    "sqft_basement",
    "lat",
    "long",
    "zipcode",
    "bedrooms",
    "sqft_above",
    "bathrooms",
    "waterfront_not provided",
    "waterfront_NO",
    "floors",
    "sqft_living15",
    "sqft_living",
]
# the independent variables set
X = data.iloc[:, ~np.isin(list(data.columns), exclude_cols)]

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# calculating VIF for each feature
vif_data["VIF"] = [vif(X.values, i) for i in range(len(X.columns))]

print(vif_data)
```

              feature       VIF
    0        sqft_lot  2.344629
    1            view  4.423597
    2       condition  7.373907
    3           grade  8.926983
    4      sqft_lot15  2.526415
    5  waterfront_YES  1.167262
    

## Simple Linear Regression Model
We'll first start with a simple linear regression model. The feature of choice will be sqft_living due to its high positive correlation with the target variable (price)

For our models we'll be using R-squared as well as Mean Absolute Error(MAE) as metrics to evaluate our models by. MAE because unlike other metrics its not as sentive to ouliers
### Findings
- The coefficient of 280.863 indicates that for every unit increase in the independent variable, the dependent variable is expected to increase by approximately 280.863 units, assuming all other factors remain constant. The intercept of -43988.892 suggests that when the independent variable has a value of zero, the predicted value of the dependent variable is approximately -43988.892.

- The R-squared value of 0.492687 suggests that approximately 49.27% of the variance in the dependent variable can be explained by the independent variable included in the model. This indicates a moderate level of goodness of fit, suggesting that the model captures a significant portion of the relationship between the variables.

- Additionally, the mean absolute error (MAE) of $173824.88 calculates the average of the absolute differences between the predicted values and the actual values.. A higher MSE implies a greater level of prediction error, and in this case, the relatively large MSE suggests that there is some degree of variability that is not captured by the model.

- Overall, while the model demonstrates a statistically significant relationship between the independent and dependent variables, there is still room for improvement in explaining the remaining variance and reducing the prediction error.

- When ouliers are removed the R-squre value decreases to 0.316, which suggests that the model is worse that the model with ouliers. However the mae does decrese to $145964.07 which is still a lot, the model might be sliightly better, but can still be improved upon.








```python
# select features
X = data["sqft_living"]
# select target
y = data["price"]

# Add a constant term to the input features
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the model summary
print(results.summary())

# Get the coefficients
coefficients = results.params
print("Coefficients:", coefficients)

# Get mean absolute error
y_pred = results.predict(X)
y_true = y
mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
print("Mean Squared Error: ", mae)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.493
    Model:                            OLS   Adj. R-squared:                  0.493
    Method:                 Least Squares   F-statistic:                 2.097e+04
    Date:                Fri, 07 Jul 2023   Prob (F-statistic):               0.00
    Time:                        23:46:11   Log-Likelihood:            -3.0006e+05
    No. Observations:               21597   AIC:                         6.001e+05
    Df Residuals:                   21595   BIC:                         6.001e+05
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const       -4.399e+04   4410.023     -9.975      0.000   -5.26e+04   -3.53e+04
    sqft_living   280.8630      1.939    144.819      0.000     277.062     284.664
    ==============================================================================
    Omnibus:                    14801.942   Durbin-Watson:                   1.982
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           542662.604
    Skew:                           2.820   Prob(JB):                         0.00
    Kurtosis:                      26.901   Cond. No.                     5.63e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.63e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    Coefficients: const         -43988.892194
    sqft_living      280.863014
    dtype: float64
    Mean Squared Error:  173824.88749617484
    

### Dealing with outliers
We'll consider any value 1.5 standard deviations away from the mean as an outlier

We'll be removing ouliers from our models this is because outliers can have a negative impact on the accuracy and reliability of a linear regression model, as they can distort the slope and intercept of the best-fit line. 


```python
# create a copy of the original data
model_data = data[["sqft_living", "price"]]

# Get mean and standard-deviation
mu = model_data["sqft_living"].mean()
sigma = model_data["sqft_living"].std()

# Compute the z-scores for each row in the 'sqft_living' column
model_data.loc[:, "sqft_living_z_score"] = (model_data["sqft_living"] - mu) / sigma

# Select only the rows where the absolute value of the z-score is less than or equal to 3
filtered_model_data = model_data[np.abs(model_data["sqft_living_z_score"]) <= 1.5]

# Drop the 'sqft_living_z_score' column
filtered_model_data = filtered_model_data.drop("sqft_living_z_score", axis=1)

# select features
X = filtered_model_data["sqft_living"]
# select target
y = filtered_model_data["price"]

# Add a constant term to the input features
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the model summary
print(results.summary())

# Get the coefficients
coefficients = results.params
print("Coefficients:", coefficients)

# Get mean absolute error
y_pred = results.predict(X)
y_true = y
mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
print("Mean Squared Error: ", mae)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.316
    Model:                            OLS   Adj. R-squared:                  0.316
    Method:                 Least Squares   F-statistic:                     9135.
    Date:                Fri, 07 Jul 2023   Prob (F-statistic):               0.00
    Time:                        23:46:45   Log-Likelihood:            -2.6959e+05
    No. Observations:               19784   AIC:                         5.392e+05
    Df Residuals:                   19782   BIC:                         5.392e+05
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        8.677e+04   4393.421     19.749      0.000    7.82e+04    9.54e+04
    sqft_living   207.4770      2.171     95.576      0.000     203.222     211.732
    ==============================================================================
    Omnibus:                     7622.888   Durbin-Watson:                   1.959
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            51231.459
    Skew:                           1.703   Prob(JB):                         0.00
    Kurtosis:                      10.110   Cond. No.                     6.24e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 6.24e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    Coefficients: const          86767.079012
    sqft_living      207.477003
    dtype: float64
    Mean Squared Error:  145964.0783632032
    

    C:\Users\Admin\AppData\Local\Temp\ipykernel_11196\3749632569.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      model_data.loc[:, 'sqft_living_z_score'] = (model_data['sqft_living'] - mu) / sigma
    

## Multiple Linear Regression Model
We'll now add on more feature to see if we can better predict the price

The features that we'll be adding to our model are condition and grade. House condition reflects the quality and maintenance of the house, which influences its value and attractiveness to buyers. Grade represents the construction and design, which also affects its desirability and demand. Both variables have a positive correlation with the house price, as shown by the bar plots. There are also the better balanced categorical features in our dataset. Therefore, we include them in our model to explain the variation in the house price.
 


```python
# list of features
features = ["sqft_living", "condition", "grade"]
# select features
X = data[features]
# select target
y = data["price"]

# Add a constant term to the input features
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the model summary
print(results.summary())

# Get the coefficients
coefficients = results.params
print("Coefficients:\n", coefficients)

# Get mean absolute error
y_pred = results.predict(X)
y_true = y
mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
print("Mean Squared Error: ", mae)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.547
    Model:                            OLS   Adj. R-squared:                  0.547
    Method:                 Least Squares   F-statistic:                     8696.
    Date:                Fri, 07 Jul 2023   Prob (F-statistic):               0.00
    Time:                        23:46:53   Log-Likelihood:            -2.9884e+05
    No. Observations:               21597   AIC:                         5.977e+05
    Df Residuals:                   21593   BIC:                         5.977e+05
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const       -4.873e+05   1.04e+04    -47.030      0.000   -5.08e+05   -4.67e+05
    sqft_living   178.4254      2.843     62.752      0.000     172.852     183.999
    condition     6.39e+04   2623.363     24.359      0.000    5.88e+04     6.9e+04
    grade        1.079e+05   2245.536     48.031      0.000    1.03e+05    1.12e+05
    ==============================================================================
    Omnibus:                    17287.572   Durbin-Watson:                   1.984
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1109737.195
    Skew:                           3.379   Prob(JB):                         0.00
    Kurtosis:                      37.461   Cond. No.                     1.44e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.44e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    Coefficients:
     const         -487265.706180
    sqft_living       178.425402
    condition       63902.630438
    grade          107856.365840
    dtype: float64
    Mean Squared Error:  161654.60391844274
    

### multiple linear regression model but with ouliers in the sqft_living removed


```python
# create a copy of the original data
model_data = data[["sqft_living", "condition", "grade", "price"]]

# Get mean and standard-deviation
mu = model_data["sqft_living"].mean()
sigma = model_data["sqft_living"].std()

# Compute the z-scores for each row in the 'sqft_living' column
model_data.loc[:, "sqft_living_z_score"] = (model_data["sqft_living"] - mu) / sigma

# Select only the rows where the absolute value of the z-score is less than or equal to 3
filtered_model_data = model_data[np.abs(model_data["sqft_living_z_score"]) <= 1.5]

# Drop the 'sqft_living_z_score' column
filtered_model_data = filtered_model_data.drop("sqft_living_z_score", axis=1)

# select features
X = filtered_model_data[features]
# select target
y = filtered_model_data["price"]


# Add a constant term to the features
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the model summary
print(results.summary())

# Get the coefficients
coefficients = results.params
print("Coefficients:\n", coefficients)

# Get mean absolute error
y_pred = results.predict(X)
y_true = y
mae = mean_absolute_error(y_pred=y_pred, y_true=y_true)
print("Mean Squared Error: ", mae)
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.422
    Model:                            OLS   Adj. R-squared:                  0.422
    Method:                 Least Squares   F-statistic:                     4817.
    Date:                Fri, 07 Jul 2023   Prob (F-statistic):               0.00
    Time:                        23:47:00   Log-Likelihood:            -2.6792e+05
    No. Observations:               19784   AIC:                         5.358e+05
    Df Residuals:                   19780   BIC:                         5.359e+05
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const        -3.26e+05   8516.214    -38.285      0.000   -3.43e+05   -3.09e+05
    sqft_living   105.9072      2.681     39.498      0.000     100.652     111.163
    condition    5.541e+04   2029.748     27.298      0.000    5.14e+04    5.94e+04
    grade        1.052e+05   1829.229     57.495      0.000    1.02e+05    1.09e+05
    ==============================================================================
    Omnibus:                     7577.882   Durbin-Watson:                   1.954
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            52688.281
    Skew:                           1.679   Prob(JB):                         0.00
    Kurtosis:                      10.256   Cond. No.                     1.35e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.35e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    Coefficients:
     const         -326043.119250
    sqft_living       105.907164
    condition       55407.623359
    grade          105170.797255
    dtype: float64
    Mean Squared Error:  132474.56426255958
    

    C:\Users\Admin\AppData\Local\Temp\ipykernel_11196\3237884997.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      model_data.loc[:, 'sqft_living_z_score'] = (model_data['sqft_living'] - mu) / sigma
    

# Regression Results
In the linear regression modeling part, four models were built to predict house prices based on different attributes of the house.

The first model, a simple linear regression, achieved an R-squared value of 0.493, indicating that 49.3% of the variance in house prices can be explained by the square footage of the living area. The model's coefficients suggest that for every square foot increase in living area, the house price is estimated to increase by $280.86. However, the model's performance can be improved as indicated by the relatively high mean absolute error of $173824.88.

The second model, a simple linear regression with outliers removed, shows a lower R-squared value of 0.316, suggesting that the removal of outliers reduced the model's explanatory power. The coefficients indicate that for every square foot increase in living area, the house price is estimated to increase by $207.48. The mean absolute error improved to $145964.07, indicating better accuracy compared to the first model.

The third model, a multiple linear regression, incorporated additional attributes such as condition and grade. It achieved an R-squared value of 0.547, indicating that 54.7% of the variance in house prices can be explained by the combined effect of square footage, condition, and grade. The coefficients suggest that the condition and grade of a house have significant impacts on its price. However, the model's mean absolute error is relatively high at $161654.60.

The fourth model, a multiple linear regression with outliers in square footage removed, shows an improved R-squared value of 0.422 compared to the third model, after removing outliers. The coefficients indicate that the condition and grade still have significant impacts on house prices. The mean absolute error decreased to $132474.56, indicating better accuracy compared to both the third model and the first two models.

Considering the model metrics, the fourth model, the multiple linear regression with outliers in square footage removed, appears to be the best model for predicting house prices. It achieved a reasonably high R-squared value, indicating a good level of explanation for the variance in house prices, while also exhibiting a lower mean absolute error compared to the other models.
